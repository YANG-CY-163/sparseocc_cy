import math
import os
import random
import mmcv
import glob
import torch
import numpy as np
from tqdm import tqdm
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from nuscenes.eval.common.utils import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from torch.utils.data import DataLoader
from models.utils import sparse2dense
from .ray_metrics import main_rayiou, main_raypq
from .ego_pose_dataset import EgoPoseDataset
from configs.r50_nuimg_704x256_8f import occ_class_names as occ3d_class_names
from configs.r50_nuimg_704x256_8f_openocc import occ_class_names as openocc_class_names
from mmcv.parallel import DataContainer as DC

@DATASETS.register_module()
class NuSceneOcc(NuScenesDataset):    
    def __init__(self, occ_gt_root, seq_mode=False, seq_split_num=1, *args, **kwargs):
        super().__init__(filter_empty_gt=False, *args, **kwargs)
        self.occ_gt_root = occ_gt_root
        self.data_infos = self.load_annotations(self.ann_file)
        self.seq_mode = seq_mode

        self.token2scene = {}
        for gt_path in glob.glob(os.path.join(self.occ_gt_root, '*/*/*.npz')):
            token = gt_path.split('/')[-2]
            scene_name = gt_path.split('/')[-3]
            self.token2scene[token] = scene_name

        for i in range(len(self.data_infos)):
            scene_name = self.token2scene[self.data_infos[i]['token']]
            self.data_infos[i]['scene_name'] = scene_name

        if seq_mode:
            self.num_frame_losses = 1
            self.queue_length = 1
            self.seq_split_num = seq_split_num
            self.random_length = 0
            self._set_sequence_group_flag() # Must be called after load_annotations b/c load_annotations does sorting.

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        res = []

        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            if idx != 0 and len(self.data_infos[idx]['sweeps']) == 0:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.seq_split_num != 1:
            if self.seq_split_num == 'all':
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(range(0, 
                                bin_counts[curr_flag], 
                                math.ceil(bin_counts[curr_flag] / self.seq_split_num)))
                        + [bin_counts[curr_flag]])

                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.seq_split_num
                self.flag = np.array(new_flags, dtype=np.int64)

    def collect_sweeps(self, index, into_past=150, into_future=0):
        all_sweeps_prev = []
        curr_index = index
        while len(all_sweeps_prev) < into_past:
            curr_sweeps = self.data_infos[curr_index]['sweeps']
            if len(curr_sweeps) == 0:
                break
            all_sweeps_prev.extend(curr_sweeps)
            all_sweeps_prev.append(self.data_infos[curr_index - 1]['cams'])
            curr_index = curr_index - 1
        
        all_sweeps_next = []
        curr_index = index + 1
        while len(all_sweeps_next) < into_future:
            if curr_index >= len(self.data_infos):
                break
            curr_sweeps = self.data_infos[curr_index]['sweeps']
            all_sweeps_next.extend(curr_sweeps[::-1])
            all_sweeps_next.append(self.data_infos[curr_index]['cams'])
            curr_index = curr_index + 1

        return all_sweeps_prev, all_sweeps_next

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        import pdb; pdb.set_trace()
        queue = []
        index_list = list(range(index-self.queue_length-self.random_length+1, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[self.random_length:])
        index_list.append(index)
        prev_scene_token = None
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            
            if not self.seq_mode: # for sliding window 
                if input_dict['scene_token'] != prev_scene_token:
                    input_dict.update(dict(prev_exists=False))
                    prev_scene_token = input_dict['scene_token']
                else:
                    input_dict.update(dict(prev_exists=True))
            
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            queue.append(example)

        for k in range(self.num_frame_losses):
            if self.filter_empty_gt and \
                (queue[-k-1] is None or ~(queue[-k-1]['gt_labels_3d']._data != -1).any()):
                return None
        return self.union2one(queue)
    
    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def union2one(self, queue):
        for key in self.collect_keys:
            if key != 'img_metas':
                queue[-1][key] = DC(torch.stack([each[key].data for each in queue]), cpu_only=False, stack=True, pad_dims=None)
            else:
                queue[-1][key] = DC([each[key].data for each in queue], cpu_only=True)
        if not self.test_mode:
            for key in ['gt_bboxes_3d', 'gt_labels_3d', 'gt_depth', 'focal']:
                if key == 'gt_bboxes_3d':
                    queue[-1][key] = DC([each[key].data for each in queue], cpu_only=True)
                else:
                    if key == 'focal':
                        queue[-1][key] = DC(torch.stack([each[key].data for each in queue]), cpu_only=False, stack=True, pad_dims=None)
                    else:
                        queue[-1][key] = DC([each[key].data for each in queue], cpu_only=False)

        queue = queue[-1]
        return queue
    
    def get_data_info(self, index):
        info = self.data_infos[index]
        sweeps_prev, sweeps_next = self.collect_sweeps(index)

        ego2global_translation = info['ego2global_translation']
        ego2global_rotation = info['ego2global_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        lidar2ego_rotation = info['lidar2ego_rotation']
        ego2global_rotation_mat = Quaternion(ego2global_rotation).rotation_matrix
        lidar2ego_rotation_mat = Quaternion(lidar2ego_rotation).rotation_matrix

        input_dict = dict(
            sample_idx=info['token'],
            sweeps={'prev': sweeps_prev, 'next': sweeps_next},
            timestamp=info['timestamp'] / 1e6,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation_mat,
            lidar2ego_translation=lidar2ego_translation,
            lidar2ego_rotation=lidar2ego_rotation_mat,
            scene_token=info['scene_token']
        )

        ego2lidar = transform_matrix(lidar2ego_translation, Quaternion(lidar2ego_rotation), inverse=True)
        input_dict['ego2lidar'] = [ego2lidar for _ in range(6)]
        input_dict['occ_path'] = os.path.join(self.occ_gt_root, info['scene_name'], info['token'], 'labels.npz')

        if self.modality['use_camera']:
            img_paths = []
            img_timestamps = []
            lidar2img_rts = []

            for _, cam_info in info['cams'].items():
                img_paths.append(os.path.relpath(cam_info['data_path']))
                img_timestamps.append(cam_info['timestamp'] / 1e6)

                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T

                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

            if not self.test_mode: # for seq_mode
                prev_exists  = not (index == 0 or self.flag[index - 1] != self.flag[index])
            else:
                prev_exists = None

            input_dict.update(dict(
                img_filename=img_paths,
                img_timestamp=img_timestamps,
                lidar2img=lidar2img_rts,
                prev_exists=prev_exists
            ))

            if not self.test_mode:
                annos = self.get_ann_info(index)
                input_dict['ann_info'] = annos

        return input_dict

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        occ_gts, occ_preds, inst_gts, inst_preds, lidar_origins = [], [], [], [], []
        print('\nStarting Evaluation...')

        sample_tokens = [info['token'] for info in self.data_infos]

        for batch in DataLoader(EgoPoseDataset(self.data_infos), num_workers=8):
            token = batch[0][0]
            output_origin = batch[1]
            
            data_id = sample_tokens.index(token)
            info = self.data_infos[data_id]

            occ_path = os.path.join(self.occ_gt_root, info['scene_name'], info['token'], 'labels.npz')
            occ_gt = np.load(occ_path, allow_pickle=True)
            gt_semantics = occ_gt['semantics']

            occ_pred = occ_results[data_id]
            sem_pred = torch.from_numpy(occ_pred['sem_pred'])  # [B, N]
            occ_loc = torch.from_numpy(occ_pred['occ_loc'].astype(np.int64))  # [B, N, 3]
            
            data_type = self.occ_gt_root.split('/')[-1]
            if data_type == 'occ3d' or data_type == 'occ3d_panoptic':
                occ_class_names = occ3d_class_names
            elif data_type == 'openocc_v2':
                occ_class_names = openocc_class_names
            else:
                raise ValueError
            free_id = len(occ_class_names) - 1
            
            occ_size = list(gt_semantics.shape)
            sem_pred, _ = sparse2dense(occ_loc, sem_pred, dense_shape=occ_size, empty_value=free_id)
            sem_pred = sem_pred.squeeze(0).numpy()

            if 'pano_inst' in occ_pred.keys():
                pano_inst = torch.from_numpy(occ_pred['pano_inst'])
                pano_sem = torch.from_numpy(occ_pred['pano_sem'])

                pano_inst, _ = sparse2dense(occ_loc, pano_inst, dense_shape=occ_size, empty_value=0)
                pano_sem, _ = sparse2dense(occ_loc, pano_sem, dense_shape=occ_size, empty_value=free_id)
                pano_inst = pano_inst.squeeze(0).numpy()
                pano_sem = pano_sem.squeeze(0).numpy()
                sem_pred = pano_sem

                gt_instances = occ_gt['instances']
                inst_gts.append(gt_instances)
                inst_preds.append(pano_inst)

            lidar_origins.append(output_origin)
            occ_gts.append(gt_semantics)
            occ_preds.append(sem_pred)
        
        if len(inst_preds) > 0:
            results = main_raypq(occ_preds, occ_gts, inst_preds, inst_gts, lidar_origins, occ_class_names=occ_class_names)
            results.update(main_rayiou(occ_preds, occ_gts, lidar_origins, occ_class_names=occ_class_names))
            return results
        else:
            return main_rayiou(occ_preds, occ_gts, lidar_origins, occ_class_names=occ_class_names)

    def format_results(self, occ_results, submission_prefix, **kwargs):
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]
            sample_token = info['token']
            save_path = os.path.join(submission_prefix, '{}.npz'.format(sample_token))
            np.savez_compressed(save_path, occ_pred.astype(np.uint8))
        
        print('\nFinished.')
