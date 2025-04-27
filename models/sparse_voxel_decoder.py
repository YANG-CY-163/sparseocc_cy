import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmcv.cnn.bricks.transformer import FFN
from .sparsebev_transformer import MultiheadFlashAttention, SparseBEVSelfAttention, SparseBEVSampling, AdaptiveMixing, TimeAdaptiveAttention
from .utils import DUMP, generate_grid, batch_indexing, history_coord_warp
from .bbox.utils import encode_bbox
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import MultiheadAttention, FFN


def index2point(coords, pc_range, voxel_size):
    """
    coords: [B, N, 3], int
    pc_range: [-40, -40, -1.0, 40, 40, 5.4]
    voxel_size: float
    """
    coords = coords * voxel_size
    coords = coords + torch.tensor(pc_range[:3], device=coords.device)
    return coords


def point2bbox(coords, box_size):
    """
    coords: [B, N, 3], float
    box_size: float
    """
    wlh = torch.ones_like(coords.float()) * box_size
    bboxes = torch.cat([coords, wlh], dim=-1)  # [B, N, 6]
    return bboxes


def upsample(pre_feat, pre_coords, interval):
    '''
    :param pre_feat: (Tensor), features from last level, (B, N, C)
    :param pre_coords: (Tensor), coordinates from last level, (B, N, 3) (3: x, y, z)
    :param interval: interval of voxels, interval = scale ** 2
    :param num: 1 -> 8
    :return: up_feat : upsampled features, (B, N*8, C//8)
    :return: up_coords: upsampled coordinates, (B, N*8, 3)
    '''
    pos_list = [0, 1, 2, [0, 1], [0, 2], [1, 2], [0, 1, 2]]
    bs, num_query, num_channels = pre_feat.shape
    
    up_feat = pre_feat.reshape(bs, num_query, 8, num_channels // 8)  # [B, N, 8, C/8]
    up_coords = pre_coords.unsqueeze(2).repeat(1, 1, 8, 1).contiguous()  # [B, N, 8, 3]
    for i in range(len(pos_list)):
        up_coords[:, :, i + 1, pos_list[i]] += interval

    up_feat = up_feat.reshape(bs, -1, num_channels // 8)
    up_coords = up_coords.reshape(bs, -1, 3)

    return up_feat, up_coords


class SparseVoxelDecoder(BaseModule):
    def __init__(self,
                 embed_dims=None,
                 num_layers=None,
                 num_frames=None,
                 num_points=None,
                 num_groups=None,
                 num_levels=None,
                 num_classes=None,
                 semantic=False,
                 topk_training=None,
                 topk_testing=None,
                 pc_range=None,
                 memory_config=dict(),):
        super().__init__()

        self.embed_dims = embed_dims
        self.num_frames = num_frames
        self.num_layers = num_layers
        self.pc_range = pc_range
        self.semantic = semantic
        self.voxel_dim = [200, 200, 16]
        self.topk_training = topk_training
        self.topk_testing = topk_testing

        self.decoder_layers = nn.ModuleList()
        self.lift_feat_heads = nn.ModuleList()
        #self.occ_pred_heads = nn.ModuleList()
        
        if semantic:
            self.seg_pred_heads = nn.ModuleList()

        for i in range(num_layers):
            self.decoder_layers.append(SparseVoxelDecoderLayer(
                 embed_dims=embed_dims,
                 num_frames=num_frames,
                 num_points=num_points // (2 ** i),
                 num_groups=num_groups,
                 num_levels=num_levels,
                 pc_range=pc_range,
                 self_attn=i in [0, 1]
            ))
            self.lift_feat_heads.append(nn.Sequential(
                nn.Linear(embed_dims, embed_dims * 8),
                nn.ReLU(inplace=True)
            ))
            #self.occ_pred_heads.append(nn.Linear(embed_dims, 1))

            if semantic:
                self.seg_pred_heads.append(nn.Linear(embed_dims, num_classes))

        # init memory
        self.num_history = memory_config['num_history']
        self.max_time_interval = memory_config['max_time_interval']
        self.memory_len = memory_config['memory_len']
        self.len_per_frame = memory_config['len_per_frame']
        self.interval = memory_config['interval']
        
        self.metas = None
        self.mask = None

        # Initialize memory banks: List[List[Dict]]
        # Outer list: layers, Inner list: time steps (FIFO queue)
        self.memory_banks = [[] for _ in range(num_layers)]

        self.padding_bbox = nn.Embedding(self.num_history, 3)  # (x, y, z)

    @torch.no_grad()
    def init_weights(self):
        for i in range(len(self.decoder_layers)):
            self.decoder_layers[i].init_weights()

    def forward(self, mlvl_feats, img_metas, reset_memory=False):
        # TODO reset memory in sparseocc.py
        if reset_memory:
            self.memory_banks = [[] for _ in range(self.num_layers)]
            self.metas = None

        occ_preds = []
        
        topk = self.topk_training if self.training else self.topk_testing
        
        B = len(img_metas)
        # init query coords
        interval = 2 ** self.num_layers
        query_coord = generate_grid(self.voxel_dim, interval).expand(B, -1, -1)  # [B, N, 3]
        query_feat = torch.zeros([B, query_coord.shape[1], self.embed_dims], device=query_coord.device)  # [B, N, C]

        for i, layer in enumerate(self.decoder_layers):
            DUMP.stage_count = i
            
            interval = 2 ** (self.num_layers - i)  # 8 4 2 1

            # bbox from coords
            query_bbox_point = index2point(query_coord, self.pc_range, voxel_size=0.4)  # [B, N, 3]
            query_bbox = point2bbox(query_bbox_point, box_size=0.4 * interval)  # [B, N, 6]
            query_bbox = encode_bbox(query_bbox, pc_range=self.pc_range)  # [B, N, 6]

            # warp history query_bbox to current frame
            temp_query_feat, temp_query_points = self.temporal_warp(img_metas, i)
            temp_query_bbox = None
            if temp_query_points is not None:
                temp_query_bbox = encode_bbox(temp_query_points, pc_range=self.pc_range)  # [B, N, 6]

            # transformer layer
            query_feat = layer(query_feat, query_bbox, temp_query_feat, temp_query_bbox, mlvl_feats, img_metas)  # [B, N, C]

            # NOTE: update the memory bank using un-normalized points
            self.update_memory(query_feat, query_bbox_point, i, timestamp=img_metas[0]['timestamp'], ego_pose=img_metas[0]['ego_pose'])
            
            # upsample 2x
            query_feat = self.lift_feat_heads[i](query_feat)  # [B, N, 8C]
            query_feat_2x, query_coord_2x = upsample(query_feat, query_coord, interval // 2)

            if self.semantic:
                seg_pred_2x = self.seg_pred_heads[i](query_feat_2x)  # [B, K, CLS]
            else:
                seg_pred_2x = None

            # sparsify after seg_pred
            non_free_prob = 1 - F.softmax(seg_pred_2x, dim=-1)[..., -1]  # [B, K]
            indices = torch.topk(non_free_prob, k=topk[i], dim=1)[1]  # [B, K]

            query_coord_2x = batch_indexing(query_coord_2x, indices, layout='channel_last')  # [B, K, 3]
            query_feat_2x = batch_indexing(query_feat_2x, indices, layout='channel_last')  # [B, K, C]
            seg_pred_2x = batch_indexing(seg_pred_2x, indices, layout='channel_last')  # [B, K, CLS]

            occ_preds.append((
                torch.div(query_coord_2x, interval // 2, rounding_mode='trunc').long(),
                None,
                seg_pred_2x,
                query_feat_2x,
                interval // 2)
            )

            query_coord = query_coord_2x.detach()
            query_feat = query_feat_2x.detach()

        return occ_preds
    

    @torch.no_grad()
    def temporal_warp(self, current_metas, layer_idx):
        """
        Retrieves the most recent memory for the layer and warps its coordinates.
        Args:
            current_metas (list[dict]): Meta information for the current batch.
            layer_idx (int): The index of the current decoder layer.

        Returns:
            tuple: (temp_query_feat, temp_query_point)
                   - temp_query_feat (Tensor | None): Features from the previous time step.
                   - temp_query_point (Tensor | None): Warped coordinates from the previous time step.
        """
        if not self.memory_banks[layer_idx]:
            return None, None
        
        # Get the most recent memory entry (last element in the list)
        prev_memory = self.memory_banks[layer_idx][-1]
        prev_ego_pose = self.metas['ego_pose']

        temp_query_feat = prev_memory['feat'] # [B, K, C]
        temp_query_point_prev = prev_memory['coord'] # [B, K, 3] - Points in PREVIOUS frame's system

        curr_ego_pose_inv = current_metas[0]['ego_pose_inv']

        # Transformation: current_lidar <- global <- previous_lidar
        T_temp2cur = curr_ego_pose_inv @ prev_ego_pose # [B, 4, 4]

        temp_query_point_curr = history_coord_warp(temp_query_point_prev, T_temp2cur, self.pc_range)
        self.memory_banks[layer_idx][-1]['coord'] = temp_query_point_curr
        
        time_interval = (current_metas[0]['timestamp'] - self.metas['timestamp']).to(temp_query_feat.device)
        valid_mask = (torch.abs(time_interval) <= self.max_time_interval)
        self.mask = valid_mask

        return temp_query_feat, temp_query_point_curr
    

    def update_memory(self, query_feat, query_point, layer_idx, timestamp, ego_pose):
        """
        Updates the memory bank for the given layer with the current features and coordinates.
        Args:
            query_feat (Tensor): Features after upsampling and top-k selection [B, K, C].
            query_point (Tensor): Points after upsampling and top-k selection [B, K, 6].
            img_metas (list[dict]): Meta information for the current batch.
            layer_idx (int): The index of the current decoder layer.
        """
        if self.memory_len <= 0:
            return

        # Detach tensors before storing
        memory_entry = {
            'feat': query_feat.detach(),
            'coord': query_point.detach()
        }
        import pdb; pdb.set_trace()
        # Append new memory
        self.memory_banks[layer_idx].append(memory_entry)

        if layer_idx == 0:
            current_metas = {   
                'timestamp': timestamp,
                'ego_pose': ego_pose
            }
            self.metas = current_metas

        # Maintain memory length (FIFO)
        if len(self.memory_banks[layer_idx]) > self.memory_len:
            self.memory_banks[layer_idx].pop(0)
            


class SparseVoxelDecoderLayer(BaseModule):
    def __init__(self,
                 embed_dims=None,
                 num_frames=None,
                 num_points=None,
                 num_groups=None,
                 num_levels=None,
                 pc_range=None,
                 self_attn=True):
        super().__init__()

        self.position_encoder = nn.Sequential(
            nn.Linear(3, embed_dims), 
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
        )

        if self_attn:
            self.self_attn = SparseBEVSelfAttention(embed_dims, num_heads=8, dropout=0.1, pc_range=pc_range, scale_adaptive=True)
            self.norm1 = nn.LayerNorm(embed_dims)
        else:
            self.self_attn = None

        #self.cross_attn = TimeAdaptiveAttention(embed_dims, num_heads=8, dropout=0.1, frames=4, num_per_frame=256)
        #self.cross_attn = MultiheadAttention(embed_dims, num_heads=8, dropout=0.1, batch_first=True)
        self.cross_attn = MultiheadFlashAttention(embed_dims, num_heads=8, dropout=0.1, batch_first=True)
        
        self.sampling = SparseBEVSampling(
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_groups=num_groups,
            num_points=num_points,
            num_levels=num_levels,
            pc_range=pc_range
        )
        self.mixing = AdaptiveMixing(
            in_dim=embed_dims,
            in_points=num_points * num_frames,
            n_groups=num_groups,
            out_points=num_points * num_frames * num_groups
        )
        self.ffn = FFN(embed_dims, feedforward_channels=embed_dims * 2, ffn_drop=0.1)
        
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        self.norm_temp = nn.LayerNorm(embed_dims)

    @torch.no_grad()
    def init_weights(self):
        if self.self_attn is not None:
            self.self_attn.init_weights()
        self.sampling.init_weights()
        self.mixing.init_weights()
        self.ffn.init_weights()

    def forward(self, query_feat, query_bbox, temp_query_feat, temp_query_bbox, mlvl_feats, img_metas):
        query_pos = self.position_encoder(query_bbox[..., :3])  # [B, N, C]

        # temporal attn
        if self.training:
            if temp_query_bbox is not None:
                temp_pos = self.position_encoder(temp_query_bbox)
            else:
                temp_pos = None
        else:
            if DUMP.stage_count == 0 and temp_query_bbox is not None:
                temp_pos = self.position_encoder(temp_query_bbox)

        query_feat = self.norm_temp(self.cross_attn(query_feat, temp_query_feat,
                                                    query_pos=query_pos, key_pos=temp_pos))
        
        query_feat = query_feat + query_pos

        if self.self_attn is not None:
            query_feat = self.norm1(self.self_attn(query_bbox, query_feat))
        sampled_feat = self.sampling(query_bbox, query_feat, mlvl_feats, img_metas)
        query_feat = self.norm2(self.mixing(sampled_feat, query_feat))
        query_feat = self.norm3(self.ffn(query_feat))

        return query_feat
