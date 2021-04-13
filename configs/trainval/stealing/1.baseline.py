# 1. data
dataset_type = 'BDWIDERFaceDataset'
data_root = 'data/wider_face_bd_results/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[1, 1, 1],
    to_rgb=True,
)
size_divisor = 32

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        typename=dataset_type,
        ann_file=data_root + 'WIDER_train/train.txt',
        img_prefix=data_root + 'WIDER_train',
        min_size=1,
        offset=0,
        pipeline=[
            dict(typename='LoadImageFromFile', to_float32=True),
            dict(typename='LoadAnnotations', with_bbox=True),
            dict(
                typename='RandomSquareCrop',
                crop_choice=[0.3, 0.45, 0.6, 0.8, 1.0],
            ),
            dict(typename='RandomFlip', flip_ratio=0.5),
            dict(typename='Resize', img_scale=(640, 640), keep_ratio=False),
            dict(typename='Normalize', **img_norm_cfg),
            dict(typename='DefaultFormatBundle'),
            dict(
                typename='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore']),
        ]),
    val=dict(
        typename=dataset_type,
        ann_file=data_root + 'WIDER_val/val.txt',
        img_prefix=data_root + 'WIDER_val',
        min_size=1,
        offset=0,
        pipeline=[
            dict(typename='LoadImageFromFile'),
            dict(
                typename='MultiScaleFlipAug',
                img_scale=(1100, 1650),
                flip=False,
                transforms=[
                    dict(typename='Resize', keep_ratio=True),
                    dict(typename='RandomFlip', flip_ratio=0.0),
                    dict(typename='Normalize', **img_norm_cfg),
                    dict(typename='Pad', size_divisor=size_divisor, pad_val=0),
                    dict(typename='ImageToTensor', keys=['img']),
                    dict(typename='Collect', keys=['img'])
                ])
        ]))

# 2. model
num_classes = 1
strides = [8, 16, 32, 64, 128]
use_sigmoid = True
scales_per_octave = 3
ratios = [1.3]
num_anchors = scales_per_octave * len(ratios)

model = dict(
    typename='SingleStageDetector',
    backbone=dict(
        typename='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        # frozen_stages=1,  # TODO
        norm_cfg=dict(
            typename='BN',
            requires_grad=True),
        norm_eval=False,
        style='pytorch'),
    neck=dict(
        typename='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    head=dict(
        typename='RetinaHead',
        num_classes=num_classes,
        num_anchors=num_anchors,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        # norm_cfg=dict(typename='GN', num_groups=32, requires_grad=True),
        use_sigmoid=use_sigmoid))

# 3. engines
meshgrid = dict(
    typename='BBoxAnchorMeshGrid',
    strides=strides,
    base_anchor=dict(
        typename='BBoxBaseAnchor',
        octave_base_scale=4,
        scales_per_octave=scales_per_octave,
        ratios=ratios,
        base_sizes=strides))

bbox_coder = dict(
    typename='DeltaXYWHBBoxCoder',
    target_means=[.0, .0, .0, .0],
    target_stds=[1.0, 1.0, 1.0, 1.0])

train_engine = dict(
    typename='TrainEngine',
    model=model,
    criterion=dict(
        typename='BBoxAnchorCriterion',
        num_classes=num_classes,
        meshgrid=meshgrid,
        bbox_coder=bbox_coder,
        loss_cls=dict(
            typename='FocalLoss',
            use_sigmoid=use_sigmoid,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(typename='L1Loss', loss_weight=1.0),
        train_cfg=dict(
            assigner=dict(
                typename='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)),
    optimizer=dict(
        typename='SGD', lr=3.75e-3, momentum=0.9, weight_decay=5e-4))  # 3 GPUS

val_engine = dict(
    typename='ValEngine',
    model=model,
    meshgrid=meshgrid,
    converter=dict(
        typename='BBoxAnchorConverter',
        num_classes=num_classes,
        bbox_coder=bbox_coder,
        nms_pre=1000,
        use_sigmoid=use_sigmoid),
    num_classes=num_classes,
    test_cfg=dict(
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(typename='nms', iou_thr=0.5),
        max_per_img=100),
    use_sigmoid=use_sigmoid,
    eval_metric=None)

# 4. hooks
hooks = [
    dict(typename='OptimizerHook'),
    dict(
        typename='CosineRestartLrSchedulerHook',
        periods=[30] * 21,
        restart_weights=[1] * 21,
        warmup='linear',
        warmup_iters=500,
        warmup_ratio=1e-1,
        min_lr_ratio=1e-2),
    dict(typename='EvalHook'),
    dict(typename='SnapshotHook', interval=10),
    dict(typename='LoggerHook', interval=10)
]

# 5. work modes
modes = ['train']  # , 'val']
max_epochs = 630

# 6. checkpoint
weights = dict(filepath='torchvision://resnet50', prefix='backbone')
# optimizer = dict(filepath='workdir/retinanet_mini/epoch_3_optim.pth')
# meta = dict(filepath='workdir/retinanet_mini/epoch_3_meta.pth')

# 7. misc
seed = 0
dist_params = dict(backend='nccl')
log_level = 'INFO'
