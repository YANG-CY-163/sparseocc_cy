import os
import cv2
from matplotlib import pyplot as plt
import utils
import logging
import argparse
import importlib
import torch
import numpy as np
from tqdm import tqdm
from mmcv import Config, DictAction
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset, build_dataloader
from configs.r50_nuimg_704x256_8f import point_cloud_range as pc_range
from configs.r50_nuimg_704x256_8f import occ_size
from configs.r50_nuimg_704x256_8f_openocc import occ_class_names
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, Normalize, hsv_to_rgb

occ_class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

color_map = np.array([
    [0, 150, 245, 255],  # car                  blue
    [160, 32, 240, 255],  # truck                purple
    [135, 60, 0, 255],  # trailer              brown
    [255, 255, 0, 255],  # bus                  yellow
    [0, 255, 255, 255],  # construction_vehicle cyan
    [255, 192, 203, 255],  # bicycle              pink
    [200, 180, 0, 255],  # motorcycle           dark orange
    [255, 0, 0, 255],  # pedestrian           red
    [255, 240, 150, 255],  # traffic_cone         light yellow
    [255, 120, 50, 255],  # barrier              orangey
    [255, 0, 255, 255],  # driveable_surface    dark pink
    [175,   0,  75, 255],       # other_flat           dark red
    [75, 0, 75, 255],  # sidewalk             dark purple
    [150, 240, 80, 255],  # terrain              light green
    [230, 230, 250, 255],  # manmade              white
    [0, 175, 0, 255],  # vegetation           green
    [255, 255, 255, 255],  # free             white
], dtype=np.uint8)

def visualize_flow(flow, arrow_interval=12, arrow_width=0.01, arrow_headwidth=3, arrow_headlength=5, save_path=None):
    """
    params: flow: [height, width, 2]
    """
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    angle = np.arctan2(-flow[..., 0], flow[..., 1])  # Negate y component here
    
    # convert to [0, 2*pi]
    angle = (angle + np.pi) % (2 * np.pi)
    hue = angle / (2 * np.pi)
    max_magnitude = np.max(magnitude)
    #import pdb; pdb.set_trace()
    saturation = magnitude / max_magnitude
    value = np.ones_like(saturation)
    hsv = np.dstack((hue, saturation, value))
    rgb = hsv_to_rgb(hsv)
    
    fig, ax = plt.subplots()
    ax.imshow(rgb)
    
    # Add HSV circle legend
    # hsv_circle = plt.imread('outputs/viz_flow/hsv_circle.png')
    # # Create an inset axes in the upper right corner
    # axins = ax.inset_axes([0.75, 0.05, 0.2, 0.2])  # [x, y, width, height] in relative coordinates
    # axins.imshow(hsv_circle)
    #axins.axis('off')
    
    height, width = flow.shape[:2]
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    mask = magnitude > 0
    # interval sample
    sampled_mask = mask[::arrow_interval, ::arrow_interval]
    sampled_x = x[::arrow_interval, ::arrow_interval][sampled_mask]
    sampled_y = y[::arrow_interval, ::arrow_interval][sampled_mask]
    sampled_u = flow[::arrow_interval, ::arrow_interval, 1][sampled_mask]
    sampled_v = -flow[::arrow_interval, ::arrow_interval, 0][sampled_mask]
    # draw arrows
    ax.quiver(sampled_x, sampled_y, sampled_u, sampled_v, color='black', scale_units='xy', scale=0.2,
              width=arrow_width, headwidth=arrow_headwidth, headlength=arrow_headlength)
    ax.axis('off')
    plt.savefig(save_path)

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

# def flow_to_2d(flow, semantics, max_flow=10.0):
#     """
#     :param flow: [H, W, D, 2]
#     :param max_flow
#     :return: image
#     """
#     H, W, D, _ = flow.shape

#     free_id = len(occ_class_names) - 1
#     flow_2d = np.zeros([H, W, 2], dtype=np.float32)

#     for i in range(D):
#         flow_i = flow[..., i, :]
#         semantics_i = semantics[..., i]
#         non_free_mask = (semantics_i != free_id)
#         flow_2d[non_free_mask] = flow_i[non_free_mask]

#     # flow_2d = flow_2d / max_flow
#     # flow_2d = np.clip(flow_2d, -1.0, 1.0)
#     return flow_2d

def flow_to_2d(flow, semantics, max_flow=10.0):
    """
    :param flow: [H, W, D, 2]
    :param semantics: [H, W, D]
    :return: flow2d [H, W, 2]
    """
    H, W, D, _ = flow.shape
    free_id = len(occ_class_names) - 1
    mask = (semantics != free_id)
    import pdb; pdb.set_trace()
    flow_mag_sq = np.sum(flow**2, axis=-1)
    
    masked_mag = np.where(mask, flow_mag_sq, -np.inf)
    best_d = np.argmax(masked_mag, axis=2)
    
    h_idx, w_idx = np.indices((H, W), sparse=False)
    
    selected_flow = flow[h_idx, w_idx, best_d]
    
    flow_2d = np.zeros((H, W, 2), dtype=np.float32)
    valid_mask = np.any(mask, axis=2)
    flow_2d[valid_mask] = selected_flow[valid_mask]
    
    return flow_2d

def main():
    parser = argparse.ArgumentParser(description='Validate a detector')
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--viz-dir', required=True)
    parser.add_argument('--override', nargs='+', action=DictAction)
    parser.add_argument('--viz_gt', action='store_true')
    args = parser.parse_args()

    # parse configs
    cfgs = Config.fromfile(args.config)
    if args.override is not None:
        cfgs.merge_from_dict(args.override)

    # use val-mini for visualization
    # cfgs.data.val.ann_file = cfgs.data.val.ann_file.replace('val', 'val_mini')

    # register custom module
    importlib.import_module('models')
    importlib.import_module('loaders')

    # MMCV, please shut up
    from mmcv.utils.logging import logger_initialized
    logger_initialized['root'] = logging.Logger(__name__, logging.WARNING)
    logger_initialized['mmcv'] = logging.Logger(__name__, logging.WARNING)

    # you need one GPU
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1

    # logging
    utils.init_logging(None, cfgs.debug)
    logging.info('Using GPU: %s' % torch.cuda.get_device_name(0))

    # random seed
    logging.info('Setting random seed: 0')
    set_random_seed(0, deterministic=True)

    logging.info('Loading validation set from %s' % cfgs.data.val.data_root)
    val_dataset = build_dataset(cfgs.data.val)
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfgs.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=0,
    )

    logging.info('Creating model: %s' % cfgs.model.type)
    model = build_model(cfgs.model)
    model.cuda()
    model = MMDataParallel(model, [0])
    model.eval()

    logging.info('Loading checkpoint from %s' % args.weights)
    load_checkpoint(
        model, args.weights, map_location='cuda', strict=True,
        logger=logging.Logger(__name__, logging.ERROR)
    )

    for i, data in tqdm(enumerate(val_loader)):
        if args.viz_gt:
            sem_gt = data['voxel_semantics'][0].numpy()  # [N]
            flow_gt = data['flow_gt'][0].numpy()

            sem_viz = occ2img(sem_gt)
            cv2.imwrite(os.path.join(args.viz_dir, 'sem_gt_%04d.jpg' % i), sem_viz[..., ::-1])

            flow_2d = flow_to_2d(flow_gt, sem_gt)
            save_path = os.path.join(args.viz_dir, 'flow_gt_%04d.jpg' % i)
            visualize_flow(flow_2d, save_path=save_path)
        
        # prediction
        with torch.no_grad():
            occ_pred = model(return_loss=False, rescale=True, **data)[0]
            #import pdb; pdb.set_trace()
            sem_pred = torch.from_numpy(occ_pred['sem_pred'])[0]  # [N]
            occ_loc = torch.from_numpy(occ_pred['occ_loc'].astype(np.int64))[0]  # [N, 3]
            flow_pred = torch.from_numpy(occ_pred['flow_pred'])[0]  # [N, 2]
            
            # sparse to dense
            free_id = len(occ_class_names) - 1
            dense_pred = torch.ones(occ_size, device=sem_pred.device, dtype=sem_pred.dtype) * free_id  # [200, 200, 16]
            dense_pred[occ_loc[..., 0], occ_loc[..., 1], occ_loc[..., 2]] = sem_pred
            
            sem_pred = dense_pred.numpy()

            dense_flow_pred = torch.zeros(occ_size + [2], device=flow_pred.device, dtype=flow_pred.dtype)
            dense_flow_pred[occ_loc[..., 0], occ_loc[..., 1], occ_loc[..., 2]] = flow_pred
            flow_pred = dense_flow_pred.numpy()

            # viz
            sem_viz = occ2img(sem_pred)
            cv2.imwrite(os.path.join(args.viz_dir, 'sem_%04d.jpg' % i), sem_viz[..., ::-1])

            flow_2d = flow_to_2d(flow_pred, sem_pred)#[..., ::-1, :]
            save_path = os.path.join(args.viz_dir, 'flow_%04d.jpg' % i)
            visualize_flow(flow_2d, save_path=save_path)
        
if __name__ == '__main__':
    main()
