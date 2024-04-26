#### Hyper-parameters ####
# auto_scale_lr = dict(base_batch_size=256, enable=True)
backend_args = None

# WGM parameters not specified are identified with _paper and come from
# https://github.com/facebookresearch/detectron2/blob/afe9eb920646102f7e6bf0cd2115841cea2aca13/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py#L24
lr = 1.0e-4
weight_decay = 0.01
batch_size_per_gpu = 32
epochs = 100
optimizer = 'AdamW'
drop_rate_path = 0.1
max_iter_paper = 184375
max_iter = 118_000 / batch_size_per_gpu
lr_step_milestones_paper = [163889, 177546]
lr_step_milestones = [x / max_iter_paper * max_iter for x in lr_step_milestones_paper]
warmup_iter = 250
warmup_factor_paper = 0.001
img_size = (960, 960)

#### Model ####
checkpoint = "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth"

data_preprocessor = dict(
    type="DetDataPreprocessor",
    mean=[103.53, 116.28, 123.675],
    std=[57.375, 57.12, 58.395],
    bgr_to_rgb=True,
    batch_augments=None,
)
model = dict(
    type="vitdet",
    image_size=img_size[0],
    data_preprocessor=data_preprocessor,
    backbone=dict(
        pretrained_model_name_or_path='facebook/vit-mae-base',
        image_size=img_size[0],
        ignore_mismatched_sizes=True,
    ),
    neck=dict(
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        in_features=768,
        out_features=256,
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32]
            # strides=[4, 8, 16, 32, 64]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))

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
work_dir = "logs/vitdet"

#### Data ####
data_root = 'data/coco/'
dataset_type = 'CocoDataset'
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
val_interval = 200
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
    # dict(param_name='lr', start_lr=warmup_factor_paper * lr, end_lr=lr, end=warmup_iter, by_epoch=False, verbose=True,
    #      type='LinearWarmupScheduler'),
    # Linear learning rate warm-up scheduler
    dict(
        type='LinearLR',  # Use linear policy to warmup learning rate
        start_factor=warmup_factor_paper, # The ratio of the starting learning rate used for warmup
        by_epoch=False,  # The warmup learning rate is updated by iteration
        begin=0,  # Start from the first iteration
        end=warmup_iter),  # End the warmup at the 500th iteration
    # The main LRScheduler
    dict(
        type='MultiStepLR',  # Use multi-step learning rate policy during training
        by_epoch=False,  # The learning rate is updated by epoch
        begin=0,   # Start from the first epoch
        # end=12,
        milestones=lr_step_milestones,  # Epochs to decay the learning rate
        gamma=0.1)  # The learning rate decay ratio
]

train_cfg = dict(
    max_epochs=epochs,
    type='EpochBasedTrainLoop',
    val_interval=val_interval,
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# WGM: not info on data augmentation in the paper
train_pipeline = [
    dict(backend_args=None, type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        img_scale=img_size,
        max_cached_images=20,
        pad_val=114.0,
        random_pop=False,
        type="CachedMosaic",
    ),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            1280,
            1280,
        ),
        type="RandomResize",
    ),
    dict(
        crop_size=img_size,
        type="RandomCrop",
    ),
    dict(type="YOLOXHSVRandomAug"),
    dict(prob=0.5, type="RandomFlip"),
    dict(
        pad_val=dict(
            img=(
                114,
                114,
                114,
            )
        ),
        size=img_size,
        type="Pad",
    ),
    dict(
        img_scale=img_size,
        max_cached_images=10,
        pad_val=(
            114,
            114,
            114,
        ),
        prob=0.5,
        random_pop=False,
        ratio_range=(
            1.0,
            1.0,
        ),
        type="CachedMixUp",
    ),
    dict(type="PackDetInputs"),
]
val_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        keep_ratio=True,
        scale=img_size,
        type="Resize",
    ),
    dict(
        pad_val=dict(
            img=(
                114,
                114,
                114,
            )
        ),
        size=img_size,
        type="Pad",
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
        interval=1, type='CheckpointHook', save_best='coco/bbox_mAP_50', rule='greater'
    ),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'),
)
custom_hooks = []

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
