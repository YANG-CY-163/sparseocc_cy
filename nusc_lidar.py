import os
import glob
import torch
import numpy as np
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.splits import train, val, test
from nuscenes.utils.data_classes import LidarPointCloud

'''
import yaml

yaml_path = './nuscenes_lidar_class.yaml'
with open(yaml_path) as f:
    lidar_class_map = yaml.full_load(f)
lidar_class_map = lidar_class_map['learning_map']
'''

lidar_class_map = {
  1: 0,
  5: 0,
  7: 0,
  8: 0,
  10: 0,
  11: 0,
  13: 0,
  19: 0,
  20: 0,
  0: 0,
  29: 0,
  31: 0,
  9: 1,
  14: 2,
  15: 3,
  16: 3,
  17: 4,
  18: 5,
  21: 6,
  2: 7,
  3: 7,
  4: 7,
  6: 7,
  12: 8,
  22: 9,
  23: 10,
  24: 11,
  25: 12,
  26: 13,
  27: 14,
  28: 15,
  30: 16
}

# https://github.com/tarashakhurana/4d-occ-forecasting/blob/ff986082cd6ea10e67ab7839bf0e654736b3f4e2/test_fgbg.py#L18C5-L18C18
def get_grid_mask(points, pc_range):
    points = points.T
    mask1 = np.logical_and(pc_range[0] <= points[0], points[0] <= pc_range[3])
    mask2 = np.logical_and(pc_range[1] <= points[1], points[1] <= pc_range[4])
    mask3 = np.logical_and(pc_range[2] <= points[2], points[2] <= pc_range[5])

    mask = mask1 & mask2 & mask3

    return mask

# https://github.com/tarashakhurana/4d-occ-forecasting/blob/ff986082cd6ea10e67ab7839bf0e654736b3f4e2/data/nusc.py#L14
class MyLidarPointCloud(LidarPointCloud):
    def get_ego_mask(self):
        ego_mask = np.logical_and(
            np.logical_and(-0.8 <= self.points[0], self.points[0] <= 0.8),
            np.logical_and(-1.5 <= self.points[1], self.points[1] <= 2.5),
        )
        return ego_mask

# https://github.com/tarashakhurana/4d-occ-forecasting/blob/ff986082cd6ea10e67ab7839bf0e654736b3f4e2/data/nusc.py#L22
class nuScenesDatasetLidar(Dataset):
    def __init__(self, nusc, nusc_split, pred_dir, gt_dir, eval_pq=False):
        """
        Figure out a list of sample data tokens for training.
        """
        super(nuScenesDatasetLidar, self).__init__()

        self.nusc = nusc
        self.nusc_split = nusc_split
        self.nusc_root = self.nusc.dataroot
        self.pred_dir = pred_dir
        self.gt_dir = gt_dir
        if "eval" in self.gt_dir:
            self.gt_filepaths = self.gt_dir
        else:
            self.gt_filepaths = sorted(glob.glob(os.path.join(self.gt_dir, '*/*/*.npz')))
        self.eval_pq = eval_pq
        #self.pc_range = [-40, -40, -1.0, 40, 40, 5.4]  # in ego system 
        self.pc_range = [-40, -40-0.9858, -1.0-1.8402, 40, 40-0.9858, 3]  # ego system - > lidar system

        scenes = self.nusc.scene

        if self.nusc_split == "train":
            split_scenes = train
        elif self.nusc_split == "val":
            split_scenes = val
        else:
            split_scenes = test

        # list all sample data
        self.valid_index = []
        self.flip_flags = []
        self.scene_tokens = []
        self.sample_tokens = []
        self.sample_data_tokens = []
        self.timestamps = []

        for scene in scenes:
            if scene["name"] not in split_scenes:
                continue
            scene_token = scene["token"]
            # location
            log = self.nusc.get("log", scene["log_token"])
            # flip x axis if in left-hand traffic (singapore)
            flip_flag = True if log["location"].startswith("singapore") else False
            #
            start_index = len(self.sample_tokens)
            first_sample = self.nusc.get("sample", scene["first_sample_token"])
            sample_token = first_sample["token"]
            i = 0
            while sample_token != "":
                self.flip_flags.append(flip_flag)
                self.scene_tokens.append(scene_token)
                self.sample_tokens.append(sample_token)
                sample = self.nusc.get("sample", sample_token)
                i += 1
                self.timestamps.append(sample["timestamp"])
                sample_data_token = sample["data"]["LIDAR_TOP"]

                self.sample_data_tokens.append(sample_data_token)
                sample_token = sample["next"]
            
            end_index = len(self.sample_tokens)
             
            valid_start_index = start_index 
            valid_end_index = end_index 
            self.valid_index += list(range(valid_start_index, valid_end_index))

        assert len(self.sample_tokens) == len(self.scene_tokens) == len(self.flip_flags) == len(self.timestamps)

    def __len__(self):
        return len(self.valid_index)

    def load_labels(self, sample_data_token):
        lidarseg = self.nusc.get("lidarseg", sample_data_token)
        lidarseg_labels = np.fromfile(
            f"{self.nusc.dataroot}/{lidarseg['filename']}", dtype=np.uint8
        )

        # map seg label from 0-31 to 0-16  
        for i in range(len(lidarseg_labels)):
            lidarseg_labels[i] = lidar_class_map[lidarseg_labels[i]]  
       
        #fg_labels = np.logical_and(1 <= lidarseg_labels, lidarseg_labels <= 23)
        return lidarseg_labels

    def __getitem__(self, idx):
        ref_index = self.valid_index[idx]

        ref_sample_token = self.sample_tokens[ref_index]
        ref_scene_token = self.scene_tokens[ref_index]
        ref_sd_token = self.sample_data_tokens[ref_index]  # sample["data"]["LIDAR_TOP"]
        flip_flag = self.flip_flags[ref_index]

        # NOTE: getting output frames
        output_origin_list = []
        output_points_list = []
        output_labels_list = []

        curr_sd = self.nusc.get("sample_data", ref_sd_token)

        # load the current lidar sweep
        curr_lidar_pc = MyLidarPointCloud.from_file(
            f"{self.nusc_root}/{curr_sd['filename']}"
        )
        ego_mask = curr_lidar_pc.get_ego_mask()
        curr_lidar_pc.points = curr_lidar_pc.points[:, np.logical_not(ego_mask)]

        origin_tf = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        points_tf = np.array(curr_lidar_pc.points[:3].T, dtype=np.float32)

        # get lidar labels  
        labels = self.load_labels(ref_sd_token).astype(np.float32)[np.logical_not(ego_mask)]
        assert len(labels) == len(points_tf)

        # mask points out of pc_range
        mask = get_grid_mask(points_tf, self.pc_range)
        points_tf = points_tf[mask]
        labels = labels[mask]

        # lidar2ego
        cs_record = self.nusc.get('calibrated_sensor',
                             curr_sd['calibrated_sensor_token'])
        l2e_r = cs_record['rotation']
        l2e_t = cs_record['translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix

        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = l2e_r_mat
        lidar2ego[:3, -1] = l2e_t
        points_pad = np.ones([points_tf.shape[0], 4])  # [N, 4]
        points_pad[:, :3] = points_tf
        points_tf = np.dot(lidar2ego[:3], points_pad.T).T  # [N ,3]

        origin_tf = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # pad to [1, 4]
        origin_tf = np.dot(lidar2ego[:3], origin_tf.T).T

        # origin
        output_origin_list.append(origin_tf)

        # points
        output_points_list.append(points_tf)

        # labels
        output_labels_list.append(labels)
        
        output_origin_tensor = torch.from_numpy(np.stack(output_origin_list))  # [T, 3]
        output_points_tensor = torch.from_numpy(np.concatenate(output_points_list))  # [N, 3]
        output_labels_tensor = torch.from_numpy(np.concatenate(output_labels_list))  # [N]

        #pred_filepath = os.path.join(self.pred_dir, ref_sample_token + '.npz')
        
        if "eval" in self.gt_filepaths:
            for gt_filepath in os.listdir(self.gt_filepaths):
                if ref_sample_token in gt_filepath: 
                    sem_pred = np.load(os.path.join(self.gt_filepaths, gt_filepath), allow_pickle=True)['gt'] # .files -> ['pred', 'gt']
                    break
        else:
            for gt_filepath in self.gt_filepaths:
                if ref_sample_token in gt_filepath: 
                    sem_pred = np.load(gt_filepath, allow_pickle=True)['semantics'] # .files -> ['semantics', 'mask_lidar', 'mask_camera']
                    break

        return (
                ref_sample_token, 
                output_origin_tensor, 
                output_points_tensor, 
                output_labels_tensor, 
                sem_pred
            )