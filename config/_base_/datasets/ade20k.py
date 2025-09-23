# dataset settings
dataset_type = 'ADE20KDataset' # 数据集类名
data_root = 'D:/Interpretable_AI/transformer_attention/ADEChallengeData2016/' # 数据集路径
# data_root = '/home/yangk/datasets/ADEChallengeData2016/'  # 指定数据集（pascal_voc12）路径


# 输入模型的图像的裁剪尺寸，一般是128的倍数，越小显存的开销越少
crop_size = (512, 512)

# 训练预处理
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

# 测试预处理
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]

# TestTimeAug后处理：同一图像，不同版本的预测结果加权集成后，作为最终的预测结果
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]

# 加载训练集
train_dataloader = dict(
    batch_size=4,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/training', seg_map_path='annotations/training'),
        pipeline=train_pipeline))

# 加载验证集
val_dataloader = dict(
    batch_size=1,
    num_workers=2,  # 指定数据加载器中的子进程数量
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline))

# 加载测试集
test_dataloader = val_dataloader

# 验证评估：设置评估指标 ['mIoU','mDice','mFscore']
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

# 测试评估，注：数据集比较小时，测试集和验证集可以相同
test_evaluator = val_evaluator
