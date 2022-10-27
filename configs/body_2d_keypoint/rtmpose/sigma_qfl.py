_base_ = ['../../_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=420, val_interval=10)

# optimizer
# optim_wrapper = dict(optimizer=dict(
#     type='Adam',
#     lr=5e-4,
# ))

# # learning policy
# param_scheduler = [
#     dict(
#         type='LinearLR', begin=0, end=500, start_factor=0.001,
#         by_epoch=False),  # warm-up
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=train_cfg['max_epochs'],
#         milestones=[380, 410],
#         gamma=0.1,
#         by_epoch=True)
# ]
max_epochs = 420
base_lr = 4e-3

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

randomness = dict(seed=42)

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=1024)

# codec settings
codec = dict(
    type='SimCCRLELabel',
    input_size=(192, 256),
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=True)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='MobileNetV2',
        widen_factor=1.,
        out_indices=(6, ),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmpose/top_down/'
            'mobilenetv2/mobilenetv2_coco_256x192-d1e58e7b_20200727.pth',
        )),
    head=dict(
        type='Sigma_Head',
        in_channels=320,
        out_channels=17,
        input_size=codec['input_size'],
        in_featuremap_size=(6, 8),
        simcc_split_ratio=codec['simcc_split_ratio'],
        deconv_out_channels=None,
        use_hilbert_flatten=True,
        use_dropout=False,
        rdrop=False,
        refine=False,
        coord_gau=False,
        hidden_dims=256,
        num_enc=1,
        dlinear=False,
        individual=False,
        s=128,
        shift=True,
        attn='laplacian',
        loss=dict(type='QFL', use_target_weight=True, size_average=True),
        decoder=codec),
    test_cfg=dict(flip_test=True, ))

# base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = '/mnt/lustre/share_data/openmmlab/datasets/detection/coco/'

file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        f'{data_root}': 's3://openmmlab/datasets/detection/coco/',
        f'{data_root}': 's3://openmmlab/datasets/detection/coco/'
    }))

# pipelines
train_pipeline = [
    dict(type='LoadImage', file_client_args=file_client_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        scale_factor=[0.75, 1.25],
        rotate_factor=60),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='Cutout', radius_factor=0.2),
    # dict(
    #     type='Albumentation',
    #     transforms=[
    #         dict(
    #             type='CoarseDropout',
    #             p=1.0,
    #             max_holes=1,
    #             max_height=codec['input_size'][1]*0.4,
    #             max_width=codec['input_size'][0]*0.4,
    #             min_height=codec['input_size'][1]*0.2,
    #             min_width=codec['input_size'][0]*0.2,
    #         )
    #     ],
    #     ),
    dict(
        type='GenerateTarget',
        target_type='keypoint_xy_label+keypoint_label',
        encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage', file_client_args=file_client_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=128,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=64,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_val2017.json',
        bbox_file=f'{data_root}person_detection_results/'
        'COCO_val2017_detections_AP_H_56_person.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# hooks
default_hooks = dict(
    checkpoint=dict(save_best='coco/AP', rule='greater', max_keep_ckpts=1))

# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/person_keypoints_val2017.json')
test_evaluator = val_evaluator
