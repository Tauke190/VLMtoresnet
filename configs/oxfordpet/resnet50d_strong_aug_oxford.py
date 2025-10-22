# model settings
model = dict(
    type='ImageClassifier',

    backbone=dict(
        type='ResNetV1d',
        depth=50,
        num_stages=4,
        out_indices=(3, ),          # Use final stage features
        drop_path_rate=0.1,         # (Optional) regularization for small dataset
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')  # use pretrained weights
    ),

    neck=dict(type='GlobalAveragePooling'),

    head=dict(
        type='LinearClsHead',
        num_classes=25,              # Oxford Pets = 37 breeds
        in_channels=2048,
        loss=dict(
            type='CrossEntropyLoss', # correct for single-label classification
            loss_weight=1.0,
            use_sigmoid=False         # ? sigmoid ? softmax (correct for Oxford Pets)
        ),
        topk=(1, 5),                 # optional: report Top-1 and Top-5 accuracy
    ),

    # Optional but helpful for small datasets like Oxford Pets
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.2),
        dict(type='CutMix', alpha=1.0)
    ]),
)
