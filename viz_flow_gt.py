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

def visualize_flow(flow, arrow_interval=5, arrow_width=0.005, arrow_headwidth=3, arrow_headlength=5, save_path=None):
    """
    params: flow: [height, width, 2]
    """
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    angle = np.arctan2(flow[..., 1], flow[..., 0])
    
    # convert to [0, 2*pi]
    angle = (angle + 2 * np.pi) % (2 * np.pi)
    hue = angle / (2 * np.pi)
    max_magnitude = np.max(magnitude)
    saturation = magnitude / max_magnitude
    value = np.ones_like(saturation)
    hsv = np.dstack((hue, saturation, value))
    rgb = hsv_to_rgb(hsv)
    fig, ax = plt.subplots()
    ax.imshow(rgb)
    height, width = flow.shape[:2]
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    mask = magnitude > 0
    # interval sample
    sampled_mask = mask[::arrow_interval, ::arrow_interval]
    sampled_x = x[::arrow_interval, ::arrow_interval][sampled_mask]
    sampled_y = y[::arrow_interval, ::arrow_interval][sampled_mask]
    sampled_u = flow[::arrow_interval, ::arrow_interval, 0][sampled_mask]
    sampled_v = flow[::arrow_interval, ::arrow_interval, 1][sampled_mask]
    # draw arrows
    ax.quiver(sampled_x, sampled_y, sampled_u, sampled_v, color='black', scale_units='xy', scale=0.1,
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

def flow_to_2d(flow, semantics, max_flow=10.0):
    """
    :param flow: [H, W, D, 2]
    :param max_flow
    :return: image
    """
    H, W, D, _ = flow.shape

    free_id = len(occ_class_names) - 1
    flow_2d = np.zeros([H, W, 2], dtype=np.float32)
    import pdb;pdb.set_trace()

    for i in range(D):
        flow_i = flow[..., i, :]
        semantics_i = semantics[..., i]
        non_free_mask = (semantics_i != free_id)
        flow_2d[non_free_mask] = flow_i[non_free_mask]

    # flow_2d = flow_2d / max_flow
    # flow_2d = np.clip(flow_2d, -1.0, 1.0)
    return flow_2d

def main():
    parser = argparse.ArgumentParser(description='Validate a detector')
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--viz-dir', required=True)
    parser.add_argument('--override', nargs='+', action=DictAction)
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

    for i, data in tqdm(enumerate(val_loader)):
        sem_gt = data['voxel_semantics'][0]  # [N]
        flow_gt = data['flow_gt'][0]

        sem_viz = occ2img(sem_gt)
        cv2.imwrite(os.path.join(args.viz_dir, 'sem_%04d.jpg' % i), sem_viz[..., ::-1])

        flow_2d = flow_to_2d(flow_gt, sem_gt)[..., ::-1, :]
        save_path = os.path.join(args.viz_dir, 'flow_%04d.jpg' % i)
        visualize_flow(flow_2d, save_path=save_path)

if __name__ == '__main__':
    main()
