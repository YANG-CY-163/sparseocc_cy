dataset_type = 'NuSceneOcc'
dataset_root = 'data/nuscenes/'
occ_gt_root = 'data/nuscenes/occupancy'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
occ_size = [200, 200, 16]

img_norm_cfg = dict(
    mean=[123.675, 116.280, 103.530],
    std=[58.395, 57.120, 57.375],
    to_rgb=True
)

# For nuScenes we usually do 10-class detection
det_class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

occ_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False
)

batch_size = 2
num_iters_per_epoch = 28130 // (batch_size)
num_epochs = 60
checkpoint_epoch_interval = 10

queue_length = 1
num_frame_losses = 1

collect_keys=['lidar2img', 'timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv']

_dim_ = 256
_num_points_ = 4
_num_groups_ = 4
_num_layers_ = 2
_num_frames_ = 8
_num_queries_ = 100
_topk_training_ = [4000, 16000, 64000]
_topk_testing_ = [2000, 8000, 32000]

model = dict(
    type='SparseOcc',
    data_aug=dict(
        img_color_aug=True,  # Move some augmentations to GPU
        img_norm_cfg=img_norm_cfg,
        img_pad_cfg=dict(size_divisor=32)),
    use_mask_camera=False,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        with_cp=True),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=_dim_,
        num_outs=4),
    pts_bbox_head=dict(
        type='SparseOccHead',
        class_names=occ_class_names,
        embed_dims=_dim_,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        transformer=dict(
            type='SparseOccTransformer',
            embed_dims=_dim_,
            num_layers=_num_layers_,
            num_frames=_num_frames_,
            num_points=_num_points_,
            num_groups=_num_groups_,
            num_queries=_num_queries_,
            num_levels=4,
            num_classes=len(occ_class_names),
            pc_range=point_cloud_range,
            occ_size=occ_size,
            topk_training=_topk_training_,
            topk_testing=_topk_testing_,
            voxel_memory_config=dict(
                num_history=500,
                max_time_interval=2,
                memory_len=1024,
                len_per_frame=256,
                interval=1),),
        loss_cfgs=dict(
            loss_mask2former=dict(
                type='Mask2FormerLoss',
                num_classes=len(occ_class_names),
                no_class_weight=0.1,
                loss_cls_weight=2.0,
                loss_mask_weight=5.0,
                loss_dice_weight=5.0,
            ),
            loss_geo_scal=dict(
                type='GeoScalLoss',
                num_classes=len(occ_class_names),
                loss_weight=1.0
            ),
            loss_sem_scal=dict(
                type='SemScalLoss',
                num_classes=len(occ_class_names),
                loss_weight=1.0
            )
        ),
    ),
)

ida_aug_conf = {
    'resize_lim': (0.38, 0.55),
    'final_dim': (256, 704),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 900, 'W': 1600,
    'rand_flip': True,
}

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    #dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=_num_frames_ - 1),
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, classes=det_class_names, is_train=True),
    dict(type='LoadOccGTFromFile', num_classes=len(occ_class_names)),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=True),
    #dict(type='DefaultFormatBundle3D', class_names=det_class_names),
    dict(type='SparseOccFormatBundle3D', class_names=det_class_names, collect_keys=collect_keys + ['prev_exists']),
    dict(type='Collect3D', keys=['img', 'voxel_semantics', 'voxel_instances', 'instance_class_ids', 'prev_exists'] + collect_keys,  # other keys: 'mask_camera'
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img', 'img_timestamp', 'ego2lidar', 'timestamp', 'ego_pose', 'ego_pose_inv'))
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    #dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=_num_frames_ - 1, test_mode=True),
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, classes=det_class_names, is_train=False),
    dict(type='LoadOccGTFromFile', num_classes=len(occ_class_names)),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),
    #dict(type='DefaultFormatBundle3D', class_names=det_class_names),
    dict(type='SparseOccFormatBundle3D', class_names=det_class_names, collect_keys=collect_keys + ['prev_exists']),
    dict(type='Collect3D', keys=['img', 'voxel_semantics', 'voxel_instances', 'instance_class_ids'] + collect_keys,
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img', 'img_timestamp', 'ego2lidar', 'scene_token', 'timestamp', 'ego_pose', 'ego_pose_inv'))
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        data_root=dataset_root,
        occ_gt_root=occ_gt_root,
        ann_file=dataset_root + 'nuscenes_infos_train_sweep.pkl',
        pipeline=train_pipeline,
        classes=det_class_names,
        modality=input_modality,
        collect_keys=collect_keys + ['img', 'prev_exists', 'img_metas'],
        queue_length=queue_length,
        num_frame_losses=num_frame_losses,
        seq_split_num=2, # streaming video training
        seq_mode=True, # streaming video training
        test_mode=False,
        load_interval=1000
    ),
    val=dict(
        type=dataset_type,
        data_root=dataset_root,
        occ_gt_root=occ_gt_root,
        ann_file=dataset_root + 'nuscenes_infos_val_sweep.pkl',
        pipeline=test_pipeline,
        classes=det_class_names,
        modality=input_modality,
        collect_keys=collect_keys + ['img', 'img_metas'],
        queue_length=queue_length,
        test_mode=True
    ),
    test=dict(
        type=dataset_type,
        data_root=dataset_root,
        occ_gt_root=occ_gt_root,
        ann_file=dataset_root + 'nuscenes_infos_test_sweep.pkl',
        pipeline=test_pipeline,
        classes=det_class_names,
        modality=input_modality,
        test_mode=True
    ),
    shuffler_sampler=dict(type='InfiniteGroupEachSampleInBatchSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=4e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            'sampling_offset': dict(lr_mult=0.1),
        }),
    weight_decay=0.001
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    by_epoch=False,
    step=[22*num_iters_per_epoch, 24*num_iters_per_epoch],
    gamma=0.2
)
total_epochs = 24

# load pretrained weights
load_from = 'pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth'
revise_keys = [('backbone', 'img_backbone')]

# resume the last training
resume_from = None

# checkpointing
checkpoint_config = dict(interval=1, max_keep_ckpts=1)

# logging
log_config = dict(
    interval=1,
    hooks=[
        dict(type='MyTextLoggerHook', by_epoch=False, interval=1, reset_flag=True),
        #dict(type='MyTensorboardLoggerHook', interval=500, reset_flag=True)
        #dict(type='MyWandbLoggerHook', by_epoch=False, interval=500, reset_flag=True, commit=True, project_name='SparseOcc', team_name='liuhs-team')
    ]
)

# evaluation
eval_config = dict(interval=num_iters_per_epoch*checkpoint_epoch_interval, by_epoch=False)

runner = dict(type='IterBasedRunner')

# other flags
debug = False