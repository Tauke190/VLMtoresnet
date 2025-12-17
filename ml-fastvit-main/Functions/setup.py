import os
import glob
import torch
import logging

from functools import partial
from timm.utils import *
from timm.loss import *

from timm.models import (
    create_model,
    safe_model_name,
    resume_checkpoint,
    load_checkpoint,
)

from timm.data import (
    create_dataset,
    create_loader,
    resolve_data_config,
    Mixup,
    FastCollateMixup,
    AugMixDataset,
)
import clip 
from CLIP.dataloaders import aircraft as aircraft_dataloader
from CLIP.dataloaders import Food101 as food101_dataloader


from timm.data.loader import fast_collate, OrderedDistributedSampler, _worker_init


torch.backends.cudnn.benchmark = True
_logger = logging.getLogger("train")

has_native_amp = False
try:
    if getattr(torch.cuda.amp, "autocast") is not None:
        has_native_amp = True
except AttributeError:
    pass

import random

def build_imagenet_clip_text_features(clip_model, device):
    base_dir = os.path.dirname(__file__)
    base_dir = os.path.dirname(base_dir)

    classes_path = os.path.join(base_dir, "misc", "imagenet_classes.txt")
    templates_path = os.path.join(base_dir, "misc",  "imagenet_templates.txt")

    with open(classes_path, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]
    with open(templates_path, "r") as f:
        templates = [line.strip() for line in f if line.strip()]

    # Sample 5-6 random prompts from all combinations
    prompt_combinations = [
        template.format(class_name)
        for class_name in class_names
        for template in templates
    ]
    sampled_prompts = random.sample(prompt_combinations, k=6)
    print("\n[DEBUG] Example text prompts for ImageNet classes:")
    for prompt in sampled_prompts:
        print("  ", prompt)

    all_class_embeds = []
    clip_model.eval()
    with torch.no_grad():
        for cls in class_names:
            texts = [t.format(cls) for t in templates]
            text_tokens = clip.tokenize(texts).to(device)
            class_feats = clip_model.encode_text(text_tokens)
            class_feats = class_feats / class_feats.norm(dim=-1, keepdim=True)
            class_feat = class_feats.mean(dim=0)
            class_feat = class_feat / class_feat.norm()
            all_class_embeds.append(class_feat)

    return torch.stack(all_class_embeds, dim=0)  # [num_classes, dim]

def build_clip_text_features(clip_model, class_names, device, template_file):
    """Build CLIP text features for validation set class names using multiple prompt templates."""
    # Load templates from file
    with open(template_file, "r") as f:
        templates = [line.strip() for line in f if line.strip()]

    # Print some example prompts for the first few classes
    print("\n[DEBUG] Example text prompts for validation dataset classes:")
    for class_name in class_names[:3]:  # Show for first 3 classes
        for template in templates:
            print("  ", template.format(class_name))
        print("---")

    all_text_features = []
    with torch.no_grad():
        for class_name in class_names:
            texts = [template.format(class_name) for template in templates]
            text_tokens = torch.cat([clip.tokenize(t) for t in texts]).to(device)
            text_features = clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # Average features for this class over all templates
            class_feature = text_features.mean(dim=0)
            class_feature = class_feature / class_feature.norm()
            all_text_features.append(class_feature)
        all_text_features = torch.stack(all_text_features, dim=0)
    return all_text_features  # [num_classes, dim]
    
def setup_validation_zeroshot(validation_dataset, validation_root, device, template_file, num_workers=4, batch_size=64, args=None):
    if not validation_root or not os.path.isdir(validation_root):
        raise ValueError(f"Invalid validation_data_dir: {validation_root}")

    # CLIP model only for TEXT encoding
    clip_model, preprocess = clip.load("ViT-L/14", device=device, jit=False)
    clip_model.eval()

    if validation_dataset == 'fgvc_aircraft':
        dataset = aircraft_dataloader( root=validation_root, train=False, transform=preprocess)
    if validation_dataset == 'food101':
        dataset = food101_dataloader( root=validation_root, train=False, transform=preprocess)

    # pick class names from dataset
    class_names = getattr(dataset, "categories", None) or getattr(dataset, "classes", None)
    if class_names is None:
        raise RuntimeError("Aircraft dataset has no 'categories' or 'classes' attribute.")

    text_features = build_clip_text_features(
        clip_model, class_names, device=device, template_file=template_file
    )

    sampler = None
    if args.distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
        sampler = OrderedDistributedSampler(dataset)
    
    # collate_fn = fast_collate 
    collate_fn = torch.utils.data.dataloader.default_collate
    loader_class = torch.utils.data.DataLoader

    loader_args = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=partial(_worker_init, worker_seeding='all'),
        persistent_workers=True
    )

    loader = loader_class(dataset, **loader_args)

    return {
        "loader": loader,
        "text_features": text_features,  # [num_classes, dim]
        "class_names": class_names,
    }


def setup_model(args):
    extra_args = {}
    extra_args['freeze_backbone'] = args.freeze_backbone
    
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
        **extra_args
    )
    if args.num_classes is None:
        assert hasattr(
            model, "num_classes"
        ), "Model must have `num_classes` attr if not set on cmd line/config."
        args.num_classes = (
            model.num_classes
        )  # FIXME handle model default vs config num_classes more elegantly

    if args.local_rank == 0:
        _logger.info(
            f"Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}"
        )

    data_config = resolve_data_config(
        vars(args), model=model, verbose=args.local_rank == 0
    )

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, "A split of 1 makes no sense"
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.cuda()

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        assert not args.split_bn
        if has_apex and use_amp == "apex":
            # Apex SyncBN preferred unless native amp is activated
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            _logger.info(
                "Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using "
                "zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled."
            )

    if args.torchscript:
        assert not use_amp == "apex", "Cannot use APEX AMP with torchscripted model"
        assert not args.sync_bn, "Cannot use SyncBatchNorm with torchscripted model"
        model = torch.jit.script(model)

    
    return model , num_aug_splits, data_config
    
def setup_distributed(args):
    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
    args.device = "cuda:0"
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = "cuda:%d" % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info(
            "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
            % (args.rank, args.world_size)
        )
    else:
        _logger.info("Training with a single process on 1 GPUs.")
    assert args.rank >= 0
    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = "apex"
    elif args.native_amp and has_native_amp:
        use_amp = "native"
    elif args.apex_amp or args.native_amp:
        _logger.warning(
            "Neither APEX or native Torch AMP is available, using float32. "
            "Install NVIDA apex or upgrade to PyTorch 1.6"
        )
    return use_amp

def basic_setup(args):
    use_amp = setup_distributed(args)
    random_seed(args.seed, args.rank)

    if args.rank == 0:
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = "-".join(
                [
                    datetime.now().strftime("%Y%m%d-%H%M%S"),
                    safe_model_name(args.model),
                    str(data_config["input_size"][-1]),
                ]
            )
        output_dir = get_outdir(
            args.output if args.output else "./output/train", exp_name
        )
        os.makedirs(output_dir, exist_ok=True)

    return use_amp 