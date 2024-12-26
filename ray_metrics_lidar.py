import os
import cv2
import copy
import math
import glob
import argparse
import numpy as np
import torch
from torch.utils.cpp_extension import load
from tqdm import tqdm
from prettytable import PrettyTable
from torch.utils.data import DataLoader
from nusc_lidar import nuScenesDatasetLidar
from nuscenes.nuscenes import NuScenes

VIZ = False

dvr = load("dvr", sources=["local/lib/dvr/dvr.cpp", "local/lib/dvr/dvr.cu"], verbose=True, extra_cuda_cflags=['-allow-unsupported-compiler'])

_pc_range = [-40, -40, -1.0, 40, 40, 5.4]
_voxel_size = 0.4
_occ_size = [200, 200, 16]

occ_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

color_map = np.array([
    [0, 0, 0, 255],    # others
    [255, 120, 50, 255],  # barrier              orangey
    [255, 192, 203, 255],  # bicycle              pink
    [255, 255, 0, 255],  # bus                  yellow
    [0, 150, 245, 255],  # car                  blue
    [0, 255, 255, 255],  # construction_vehicle cyan
    [200, 180, 0, 255],  # motorcycle           dark orange
    [255, 0, 0, 255],  # pedestrian           red
    [255, 240, 150, 255],  # traffic_cone         light yellow
    [135, 60, 0, 255],  # trailer              brown
    [160, 32, 240, 255],  # truck                purple
    [255, 0, 255, 255],  # driveable_surface    dark pink
    [175,   0,  75, 255],       # other_flat           dark red
    [75, 0, 75, 255],  # sidewalk             dard purple
    [150, 240, 80, 255],  # terrain              light green
    [230, 230, 250, 255],  # manmade              white
    [0, 175, 0, 255],  # vegetation           green
    [255, 255, 255, 255],  # free             white
], dtype=np.uint8)


def occ2img(semantics):
    H, W, D = semantics.shape

    free_id = len(occ_class_names) - 1
    semantics_2d = np.ones([H, W], dtype=np.int32) * free_id

    for i in range(D):
        semantics_i = semantics[..., i]
        non_free_mask = (semantics_i != free_id)
        semantics_2d[non_free_mask] = semantics_i[non_free_mask]

    viz = color_map[semantics_2d]
    viz = viz[..., :3]
    viz = cv2.resize(viz, dsize=(800, 800))

    return viz


def viz_tp(pcd, cls_mask, tp_mask):
    pcd = copy.deepcopy(pcd)
    pcd[..., 0] -= _pc_range[0]
    pcd[..., 1] -= _pc_range[1]
    pcd[..., 2] -= _pc_range[2]
    pcd /= _voxel_size
    pcd = pcd.astype(np.int32)
    pcd[..., 0] = np.clip(pcd[..., 0], a_min=0, a_max=200-1)
    pcd[..., 1] = np.clip(pcd[..., 1], a_min=0, a_max=200-1)
    pcd[..., 2] = np.clip(pcd[..., 2], a_min=0, a_max=16-1)

    free_id = len(occ_class_names) - 1
    pcd_dense = np.ones([200, 200, 16], dtype=np.int32) * free_id

    pcd_dense[pcd[cls_mask][..., 0], pcd[cls_mask][..., 1], pcd[cls_mask][..., 2]] = 0
    pcd_dense[pcd[tp_mask][..., 0], pcd[tp_mask][..., 1], pcd[tp_mask][..., 2]] = 1

    return occ2img(pcd_dense)


def viz_pcd(pcd, cls):
    pcd = copy.deepcopy(pcd.astype(np.float32))
    pcd[..., 0] -= _pc_range[0]
    pcd[..., 1] -= _pc_range[1]
    pcd[..., 2] -= _pc_range[2]
    pcd[..., 0:3] /= _voxel_size
    pcd = pcd.astype(np.int32)
    pcd[..., 0] = np.clip(pcd[..., 0], a_min=0, a_max=200-1)
    pcd[..., 1] = np.clip(pcd[..., 1], a_min=0, a_max=200-1)
    pcd[..., 2] = np.clip(pcd[..., 2], a_min=0, a_max=16-1)

    free_id = len(occ_class_names) - 1
    pcd_dense = np.ones([200, 200, 16], dtype=np.int32) * free_id

    pcd_dense[pcd[..., 0], pcd[..., 1], pcd[..., 2]] = cls.astype(np.int32)

    return occ2img(pcd_dense)


# https://github.com/tarashakhurana/4d-occ-forecasting/blob/ff986082cd6ea10e67ab7839bf0e654736b3f4e2/test_fgbg.py#L29C1-L46C16
def get_rendered_pcds(origin, points, tindex, pred_dist):
    pcds = []
    for t in range(len(origin)):
        mask = (tindex == t)
        # skip the ones with no data
        if not mask.any():
            continue
        _pts = points[mask, :3]
        # use ground truth lidar points for the raycasting direction
        v = _pts - origin[t][None, :]
        d = v / np.sqrt((v ** 2).sum(axis=1, keepdims=True))
        pred_pts = origin[t][None, :] + d * pred_dist[mask][:, None]
        pcds.append(torch.from_numpy(pred_pts))
    return pcds


def meshgrid3d(occ_size, pc_range):
    W, H, D = occ_size
    
    xs = torch.linspace(0.5, W - 0.5, W).view(W, 1, 1).expand(W, H, D) / W
    ys = torch.linspace(0.5, H - 0.5, H).view(1, H, 1).expand(W, H, D) / H
    zs = torch.linspace(0.5, D - 0.5, D).view(1, 1, D).expand(W, H, D) / D
    xs = xs * (pc_range[3] - pc_range[0]) + pc_range[0]
    ys = ys * (pc_range[4] - pc_range[1]) + pc_range[1]
    zs = zs * (pc_range[5] - pc_range[2]) + pc_range[2]
    xyz = torch.stack((xs, ys, zs), -1)

    return xyz

def process_one_sample(sem_pred, output_origin, output_points, output_labels, return_xyz=False):
    # lidar origin in ego coordinate
    # lidar_origin = torch.tensor([[[0.9858, 0.0000, 1.8402]]])
    T = output_origin.shape[1]
    pred_pcds_t = []
    gt_pcds_t = []

    free_id = len(occ_class_names) - 1 
    occ_pred = copy.deepcopy(sem_pred)
    occ_pred[sem_pred < free_id] = 1
    occ_pred[sem_pred == free_id] = 0
    occ_pred = occ_pred.permute(2, 1, 0)
    occ_pred = occ_pred[None, None, :].contiguous().float()

    offset = torch.Tensor(_pc_range[:3])[None, None, :]
    scaler = torch.Tensor([_voxel_size] * 3)[None, None, :]

    lidar_tindex = torch.zeros([1, output_points.shape[1]])
    
    for t in range(T): 
        lidar_origin = output_origin[:, t:t+1, :]  # [1, 1, 3]
        lidar_endpts = output_points  # [1, N, 3]

        output_origin_render = ((lidar_origin - offset) / scaler).float()  # [1, 1, 3]
        output_points_render = ((lidar_endpts - offset) / scaler).float()  # [1, N, 3]
        output_tindex_render = lidar_tindex  # [1, N], all zeros

        with torch.no_grad():
            pred_dist, gt_dist, coord_index = dvr.render_forward(
                occ_pred.cuda(),
                output_origin_render.cuda(),
                output_points_render.cuda(),
                output_tindex_render.cuda(),
                [1, 16, 200, 200],
                "test"
            )
            pred_dist *= _voxel_size
            gt_dist *= _voxel_size

        pred_pcds = get_rendered_pcds(
            lidar_origin[0].cpu().numpy(),
            lidar_endpts[0].cpu().numpy(),
            lidar_tindex[0].cpu().numpy(),
            pred_dist[0].cpu().numpy()
        )
        gt_pcds = get_rendered_pcds(
            lidar_origin[0].cpu().numpy(),
            lidar_endpts[0].cpu().numpy(),
            lidar_tindex[0].cpu().numpy(),
            gt_dist[0].cpu().numpy()
        )
        coord_index = coord_index[0, :, :].long().cpu()  # [N, 3]

        pred_label = sem_pred[coord_index[:, 0], coord_index[:, 1], coord_index[:, 2]][:, None]  # [N, 1]
        pred_dist = pred_dist[0, :, None].cpu()

        gt_label = output_labels[0, :, None].cpu()  # [N, 1]
        gt_dist = gt_dist[0, :, None].cpu()

        if return_xyz:
            pred_pcds = torch.cat([pred_label, pred_dist, pred_pcds[0]], dim=-1)  # [N, 5]  5: [label, dist, x, y, z]
            gt_pcds = torch.cat([gt_label, gt_dist, gt_pcds[0]], dim=-1)  # [N, 5]  5: [label, dist, x, y, z]

        else:
            pred_pcds = torch.cat([pred_label, pred_dist], dim=-1)
            gt_pcds = torch.cat([gt_label, gt_dist], dim=-1)

        pred_pcds_t.append(pred_pcds)
        gt_pcds_t.append(gt_pcds)

    pred_pcds_t = torch.cat(pred_pcds_t, dim=0)
    gt_pcds_t = torch.cat(gt_pcds_t, dim=0)
   
    return pred_pcds_t.numpy(), gt_pcds_t.numpy()


def calc_metrics(pcd_pred_list, pcd_gt_list):
    thresholds = [1, 2, 4]

    gt_cnt = np.zeros([len(occ_class_names)])
    pred_cnt = np.zeros([len(occ_class_names)])
    tp_cnt = np.zeros([len(thresholds), len(occ_class_names)])

    count = 0

    for pcd_pred, pcd_gt in tqdm(zip(pcd_pred_list, pcd_gt_list), ncols=50):
        for j, threshold in enumerate(thresholds):
            # L1
            depth_pred = pcd_pred[:, 1]
            depth_gt = pcd_gt[:, 1]
            l1_error = np.abs(depth_pred - depth_gt)
            tp_dist_mask = l1_error < threshold
            
            for i, cls in enumerate(occ_class_names):
                cls_id = occ_class_names.index(cls)
                cls_mask_pred = (pcd_pred[:, 0].astype(np.int32) == cls_id)
                cls_mask_gt = (pcd_gt[:, 0].astype(np.int32) == cls_id)

                gt_cnt_i = cls_mask_gt.sum()
                pred_cnt_i = cls_mask_pred.sum()
                if j == 0:
                    gt_cnt[i] += gt_cnt_i
                    pred_cnt[i] += pred_cnt_i

                tp_cls = cls_mask_gt & cls_mask_pred  # [N]
                tp_mask = np.logical_and(tp_cls, tp_dist_mask)
                tp_cnt[j][i] += tp_mask.sum()

        if VIZ:
            cv2.imwrite('%04d_gt.jpg' % count, viz_pcd(pcd_gt[:, 2:], pcd_gt[:, 0])[..., ::-1])
            cv2.imwrite('%04d_pd.jpg' % count, viz_pcd(pcd_pred[:, 2:], pcd_pred[:, 0])[..., ::-1])
        count += 1
    
    # print('gt_cnt', gt_cnt)
    # print('pred_cnt', pred_cnt)
    # print('TP', tp_cnt[j])

    meanIoU = []
    for j, threshold in enumerate(thresholds):
        meanIoU.append((tp_cnt[j] / (gt_cnt + pred_cnt - tp_cnt[j]))[:-1])
    meanIoU = np.array(meanIoU, dtype=np.float32)

    return meanIoU


def make_data_loader(pred_dir, gt_dir):
    data_loader_kwargs={
        "pin_memory": False,  # NOTE
        "shuffle": False,
        "batch_size": 1,
        "num_workers": 8,
    }

    nusc = NuScenes('v1.0-trainval', 'data/nuscenes')

    data_loader = DataLoader(
        nuScenesDatasetLidar(nusc, 'val', pred_dir, gt_dir),
        **data_loader_kwargs,
    )

    return data_loader


def main(args):
    
    # dataset
    data_loader = make_data_loader(args.pred_dir, args.gt_dir)

    pcd_pred_list, pcd_gt_list = [], []
    for i, batch in tqdm(enumerate(data_loader), ncols=50):
        output_origin, output_points, output_labels = batch[1:4]
        sem_pred = batch[4][0]
    
        return_xyz = VIZ
        pcd_pred, pcd_gt = process_one_sample(sem_pred, output_origin, output_points, output_labels, return_xyz=return_xyz)
        
        assert pcd_pred.shape == pcd_gt.shape
        pcd_pred_list.append(pcd_pred)
        pcd_gt_list.append(pcd_gt)
        
    iou_list = calc_metrics(pcd_pred_list, pcd_gt_list)
    
    table = PrettyTable([
        'Class Names',
        'IoU@1', 'IoU@2', 'IoU@4'
    ])
    table.float_format = '.3f'

    for i in range(len(occ_class_names) - 1):
        table.add_row([
            occ_class_names[i],
            iou_list[0][i], iou_list[1][i], iou_list[2][i]
        ], divider=(i == len(occ_class_names) - 2))
    
    table.add_row([
        'MEAN',
        np.mean(iou_list[0]), np.mean(iou_list[1]), np.mean(iou_list[2])
    ])

    print(table)
    print('# mIoU:', np.mean(iou_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", type=str)
    parser.add_argument("--gt-dir", type=str)
    args = parser.parse_args()

    torch.random.manual_seed(0)
    np.random.seed(0)

    main(args)