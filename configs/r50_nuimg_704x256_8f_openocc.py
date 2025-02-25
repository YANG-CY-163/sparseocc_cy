_base_ = ['./r50_nuimg_704x256_8f.py']

occ_gt_root = 'data/nuscenes/openocc_v2'

det_class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

occ_class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

_num_frames_ = 8
_voxel_flow_ = True
_instance_flow_ = False

model = dict(
    pts_bbox_head=dict(
        class_names=occ_class_names,
        voxel_flow=_voxel_flow_,
        instance_flow=_instance_flow_,
        transformer=dict(
            num_classes=len(occ_class_names),
            voxel_flow=_voxel_flow_,
            instance_flow=_instance_flow_),
        loss_cfgs=dict(
            loss_mask2former=dict(
                class_names=occ_class_names,
                num_classes=len(occ_class_names),
                loss_flow_cfg=dict(type='L1Loss', loss_weight=1.0),
                flow=_instance_flow_
            ),
            loss_flow=dict(type='L1Loss', loss_weight=1.0),  # TODO loss weight
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
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=_num_frames_ - 1),
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, classes=det_class_names, is_train=True),
    dict(type='LoadOccGTFromFile', num_classes=len(occ_class_names)),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=True),
    dict(type='DefaultFormatBundle3D', class_names=det_class_names),
    dict(type='Collect3D', keys=['img', 'voxel_semantics', 'voxel_instances', 'instance_class_ids', 'flow_gt'],  # other keys: 'mask_camera'
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img', 'img_timestamp', 'ego2lidar'))
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=_num_frames_ - 1, test_mode=True),
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, classes=det_class_names, is_train=False),
    dict(type='LoadOccGTFromFile', num_classes=len(occ_class_names)),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),
    dict(type='DefaultFormatBundle3D', class_names=det_class_names),
    dict(type='Collect3D', keys=['img', 'voxel_semantics', 'voxel_instances', 'instance_class_ids', 'flow_gt'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img', 'img_timestamp', 'ego2lidar'))
]

data = dict(
    workers_per_gpu=8,
    train=dict(
        pipeline=train_pipeline,
        occ_gt_root=occ_gt_root
    ),
    val=dict(
        pipeline=test_pipeline,
        occ_gt_root=occ_gt_root
    ),
    test=dict(
        pipeline=test_pipeline,
        occ_gt_root=occ_gt_root
    ),
)