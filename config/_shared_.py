#### Hyper-parameters ####
auto_scale_lr = dict(base_batch_size=256, enable=True)
backend_args = None

lr = 0.004
weight_decay = 0.05
batch_size_per_gpu = 80
epochs = 300
stage2_num_epochs = 20
optimizer = "AdamW"

#### Data ####
data_root = "../coco/"
dataset_type = "CocoDataset"
img_scales = [
    (
        640,
        640,
    ),
    (
        320,
        320,
    ),
    (
        960,
        960,
    ),
]
load_from = None
data_preprocessor = dict(
    type="DetDataPreprocessor",
    mean=[103.53, 116.28, 123.675],
    std=[57.375, 57.12, 58.395],
    bgr_to_rgb=False,
    batch_augments=None,
)


#### Training/Evaluation ####
metric = "mAP"
evaluation = dict(save_best=metric)
val_interval = 5
val_ann_file = "annotations/instances_val2017.json"
val_img_prefix = "val2017/"
train_ann_file = "annotations/instances_train2017.json"
train_img_prefix = "train2017/"

optim_wrapper = dict(
    loss_scale="dynamic",
    optimizer=dict(lr=lr, type=optimizer, weight_decay=weight_decay),
    paramwise_cfg=dict(bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type="AmpOptimWrapper",
)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(begin=0, by_epoch=False, end=1000, start_factor=1e-05, type="LinearLR"),
    dict(
        T_max=epochs // 2,
        begin=epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True,
        end=epochs,
        eta_min=lr * 0.05,
        type="CosineAnnealingLR",
    ),
]

train_cfg = dict(
    dynamic_intervals=[
        (
            epochs - stage2_num_epochs,
            1,
        ),
    ],
    max_epochs=epochs,
    type="EpochBasedTrainLoop",
    val_interval=val_interval,
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")


train_pipeline = [
    dict(backend_args=None, type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        img_scale=(
            640,
            640,
        ),
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
        crop_size=(
            640,
            640,
        ),
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
        size=(
            640,
            640,
        ),
        type="Pad",
    ),
    dict(
        img_scale=(
            640,
            640,
        ),
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
train_pipeline_stage2 = [
    dict(backend_args=None, type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            640,
            640,
        ),
        type="RandomResize",
    ),
    dict(
        crop_size=(
            640,
            640,
        ),
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
        size=(
            640,
            640,
        ),
        type="Pad",
    ),
    dict(type="PackDetInputs"),
]
val_pipeline = [
    dict(backend_args=None, type="LoadImageFromFile"),
    dict(
        keep_ratio=True,
        scale=(
            640,
            640,
        ),
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
        size=(
            640,
            640,
        ),
        type="Pad",
    ),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
        ),
        type="PackDetInputs",
    ),
]
test_pipeline = [
    dict(backend_args=None, type="LoadImageFromFile"),
    dict(
        keep_ratio=True,
        scale=(
            640,
            640,
        ),
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
        size=(
            640,
            640,
        ),
        type="Pad",
    ),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
        ),
        type="PackDetInputs",
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
        type="CocoDataset",
    ),
    num_workers=10,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type="DefaultSampler"),
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
        type="CocoDataset",
    ),
    drop_last=False,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(shuffle=False, type="DefaultSampler"),
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
        type="CocoDataset",
    ),
    drop_last=False,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(shuffle=False, type="DefaultSampler"),
)

val_evaluator = dict(
    ann_file=data_root + val_ann_file,
    backend_args=None,
    format_only=False,
    metric="bbox",
    proposal_nums=(
        100,
        1,
        10,
    ),
    type="CocoMetric",
)
test_evaluator = val_evaluator

# Test Time Augmentation
tta_model = dict(
    tta_cfg=dict(max_per_img=100, nms=dict(iou_threshold=0.6, type="nms")),
    type="DetTTAModel",
)
tta_pipeline = [
    dict(backend_args=None, type="LoadImageFromFile"),
    dict(
        transforms=[
            [
                dict(
                    keep_ratio=True,
                    scale=(
                        640,
                        640,
                    ),
                    type="Resize",
                ),
                dict(
                    keep_ratio=True,
                    scale=(
                        320,
                        320,
                    ),
                    type="Resize",
                ),
                dict(
                    keep_ratio=True,
                    scale=(
                        960,
                        960,
                    ),
                    type="Resize",
                ),
            ],
            [
                dict(prob=1.0, type="RandomFlip"),
                dict(prob=0.0, type="RandomFlip"),
            ],
            [
                dict(
                    pad_val=dict(
                        img=(
                            114,
                            114,
                            114,
                        )
                    ),
                    size=(
                        960,
                        960,
                    ),
                    type="Pad",
                ),
            ],
            [
                dict(type="LoadAnnotations", with_bbox=True),
            ],
            [
                dict(
                    meta_keys=(
                        "img_id",
                        "img_path",
                        "ori_shape",
                        "img_shape",
                        "scale_factor",
                        "flip",
                        "flip_direction",
                    ),
                    type="PackDetInputs",
                ),
            ],
        ],
        type="TestTimeAug",
    ),
]


#### Hooks ####
default_hooks = dict(
    checkpoint=dict(
        interval=-1, type="CheckpointHook", save_best="coco/bbox_mAP_50", rule="greater"
    ),
    logger=dict(interval=50, type="LoggerHook"),
    param_scheduler=dict(type="ParamSchedulerHook"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    timer=dict(type="IterTimerHook"),
    visualization=dict(type="DetVisualizationHook"),
)
custom_hooks = [
    dict(
        ema_type="ExpMomentumEMA",
        momentum=0.0002,
        priority=49,
        type="EMAHook",
        update_buffers=True,
    ),
    dict(
        switch_epoch=epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2,
        type="PipelineSwitchHook",
    ),
]

#### Viz ####
visualizer = dict(
    type="Visualizer",
    vis_backends=[
        dict(type="LocalVisBackend"),
    ],
)

#### Settings ####
resume = False
default_scope = "mmdet"
work_dir = "../logs"
backend_args = None
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend="nccl"),
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
)
log_level = "INFO"
log_processor = dict(by_epoch=True, type="LogProcessor", window_size=50)
