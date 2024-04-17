#### Hyper-parameters ####
# auto_scale_lr = dict(base_batch_size=256, enable=True)
backend_args = None

# WGM parameters not specified are identified with _paper and come from
# https://github.com/facebookresearch/detectron2/blob/afe9eb920646102f7e6bf0cd2115841cea2aca13/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py#L24
lr = 1.0e-4
weight_decay = 0.01
batch_size_per_gpu = 80
epochs = 100
optimizer = 'AdamW'
drop_rate_path = 0.1
max_iter_paper = 184375
max_iter = 118_000 / batch_size_per_gpu
lr_step_milestones_paper = [163889, 177546]
lr_step_milestones = [x / max_iter_paper * max_iter for x in lr_step_milestones_paper]
warmup_iter = 250
warmup_factor_paper = 0.001

#### Model ####
checkpoint = "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth"

data_preprocessor = dict(
    type="DetDataPreprocessor",
    mean=[103.53, 116.28, 123.675],
    std=[57.375, 57.12, 58.395],
    bgr_to_rgb=False,
    batch_augments=None,
)
model = dict(
    type="vitdet",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='vit_b_16',
        pretrained=True,
    ),
    neck=dict(
    ),
    bbox_head=dict(
    ),
    train_cfg=dict(
    ),
    test_cfg=dict(
    ),
)

visualizer = dict(
    type="Visualizer",
    vis_backends=[
        dict(type="LocalVisBackend"),
        # dict(
        #     type="WandbVisBackend",
        #     init_kwargs=dict(project="rtmdet", entity="gif-7010"),
        # ),
    ],
)
work_dir = "../logs/vitdet"

#### Data ####
data_root = 'data/coco/'
dataset_type = 'CocoDataset'
img_size = 1024
load_from = None
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[103.53, 116.28, 123.675],
    std=[57.375, 57.12, 58.395],
    bgr_to_rgb=False,
    batch_augments=None,
)

#### Training/Evaluation ####
metric = 'mAP'
evaluation = dict(save_best=metric)
val_interval = 5
val_ann_file = 'annotations/instances_val2017.json'
val_img_prefix = 'val2017/'
train_ann_file = 'annotations/instances_train2017.json'
train_img_prefix = 'train2017/'

optim_wrapper = dict(
    loss_scale='dynamic',
    optimizer=dict(lr=lr, type=optimizer, weight_decay=weight_decay),
    paramwise_cfg=dict(bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='AmpOptimWrapper',
)
# optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

# TODO implement layer-wise learning rate https://github.com/open-mmlab/mmdetection/issues/4100#issuecomment-726660873
#      value of 0.7/0.8/0.9 for ViT-B/L/H
#      embedding_decayed_lr = learning_rate * (layer_decay ** (n_layers+1))

param_scheduler = [
    dict(start_lr=warmup_factor_paper * lr, end_lr=lr, end=warmup_iter, by_epoch=False, verbose=True, type='LinearWarmupScheduler'),
    dict(begin=warmup_iter, milestones=lr_step_milestones, by_epoch=False, verbose=True, type='MultiStepLR'),
]

train_cfg = dict(
    max_epochs=epochs,
    type='EpochBasedTrainLoop',
    val_interval=val_interval,
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        type="RandomResize",
    ),
    dict(
        crop_size=img_size,
        type="RandomCrop",
    ),
    dict(prob=0.5, direction='horizontal', type='RandomFlip'),

    dict(type='PackDetInputs'),
]
val_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        keep_ratio=True,
        scale=img_size,
        type="Resize",
    ),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs',
    ),
]
test_pipeline = val_pipeline

inference_pipeline = [
    dict(backend_args=backend_args, type='LoadImageFromFile'),
    dict(
        keep_ratio=True,
        scale=img_size,
        type='Resize',
    ),
]

train_dataloader = dict(
    batch_sampler=None,
    batch_size=batch_size_per_gpu,
    dataset=dict(
        ann_file=train_ann_file,
        backend_args=None,
        data_prefix=dict(img=train_img_prefix),
        data_root=data_root,
        filter_cfg=dict(filter_empty_gt=True, min_size=batch_size_per_gpu),
        pipeline=train_pipeline,
        type='CocoDataset',
    ),
    num_workers=10,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'),
)
val_dataloader = dict(
    batch_size=10,
    dataset=dict(
        ann_file=val_ann_file,
        backend_args=None,
        data_prefix=dict(img=val_img_prefix),
        data_root=data_root,
        pipeline=val_pipeline,
        test_mode=True,
        type='CocoDataset',
    ),
    drop_last=False,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'),
)
test_dataloader = dict(
    batch_size=10,
    dataset=dict(
        ann_file=val_ann_file,
        backend_args=None,
        data_prefix=dict(img=val_img_prefix),
        data_root=data_root,
        pipeline=test_pipeline,
        test_mode=True,
        type='CocoDataset',
    ),
    drop_last=False,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'),
)

val_evaluator = dict(
    ann_file=data_root + val_ann_file,
    backend_args=None,
    format_only=False,
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='CocoMetric',
)
test_evaluator = val_evaluator

# Test Time Augmentation
tta_model = dict(
    tta_cfg=dict(max_per_img=100, nms=dict(iou_threshold=0.6, type='nms')),
    type='DetTTAModel',
)
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(
                    keep_ratio=True,
                    scale=img_size,
                    type='Resize',
                ),
                dict(
                    keep_ratio=True,
                    scale=(
                        320,
                        320,
                    ),
                    type='Resize',
                ),
                dict(
                    keep_ratio=True,
                    scale=(
                        960,
                        960,
                    ),
                    type='Resize',
                ),
            ],
            [
                dict(prob=1.0, type='RandomFlip'),
                dict(prob=0.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations', with_bbox=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                        'flip',
                        'flip_direction',
                    ),
                    type='PackDetInputs',
                ),
            ],
        ],
        type='TestTimeAug',
    ),
]

#### Hooks ####
default_hooks = dict(
    checkpoint=dict(
        interval=-1, type='CheckpointHook', save_best='coco/bbox_mAP_50', rule='greater'
    ),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'),
)

#### Viz ####
visualizer = dict(
    type='Visualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ],
)

#### Settings ####
resume = False
default_scope = 'mmdet'
backend_args = None
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
)
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
