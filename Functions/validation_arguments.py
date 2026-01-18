import argparse
import yaml


def _parse_args():
    config_parser = argparse.ArgumentParser(
        description="Validation Config", add_help=False
    )
    # Add --config to config_parser
    config_parser.add_argument(
        "-c",
        "--config",
        default="",
        type=str,
        metavar="FILE",
        help="YAML config file specifying default arguments",
    )

    parser = argparse.ArgumentParser(description="Validation")
    
    parser.add_argument("data", metavar="DIR", help="path to dataset")

    parser.add_argument(
        "-c",
        "--config",
        default="",
        type=str,
        metavar="FILE",
        help="YAML config file specifying default arguments",
    )

    parser.add_argument(
    "--split",
    metavar="NAME",
    default="validation",
    help="dataset split (default: validation)",
    )
    parser.add_argument(
        "--dataset-download",
        action="store_true",
        default=False,
        help="Allow download of dataset for torch/ and tfds/ datasets that support it.",
    )
    parser.add_argument(
        "--model",
        "-m",
        metavar="NAME",
        default="dpn92",
        help="model architecture (default: dpn92)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256)",
    )
    parser.add_argument(
        "--img-size",
        default=None,
        type=int,
        metavar="N",
        help="Input image dimension, uses model default if empty",
    )
    parser.add_argument(
        "--input-size",
        default=[3, 256, 256],
        nargs=3,
        type=int,
        metavar="N N N",
        help="Input all image dimensions (d h w, e.g. --input-size 3 256 256), uses model default if empty",
    )
    parser.add_argument(
        "--crop-pct",
        default=None,
        type=float,
        metavar="N",
        help="Input image center crop pct",
    )
    parser.add_argument(
        "--mean",
        type=float,
        nargs="+",
        default=None,
        metavar="MEAN",
        help="Override mean pixel value of dataset",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs="+",
        default=None,
        metavar="STD",
        help="Override std deviation of of dataset",
    )
    parser.add_argument(
        "--interpolation",
        default="",
        type=str,
        metavar="NAME",
        help="Image resize interpolation type (overrides model)",
    )
    parser.add_argument(
        "--class-map",
        default="",
        type=str,
        metavar="FILENAME",
        help='path to class to idx mapping file (default: "")',
    )
    parser.add_argument(
        "--gp",
        default=None,
        type=str,
        metavar="POOL",
        help="Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.",
    )
    parser.add_argument(
        "--log-freq",
        default=10,
        type=int,
        metavar="N",
        help="batch logging frequency (default: 10)",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--pretrained", dest="pretrained", action="store_true", help="use pre-trained model"
    )
    parser.add_argument("--num-gpu", type=int, default=1, help="Number of GPUS to use")
    parser.add_argument(
        "--test-pool", dest="test_pool", action="store_true", help="enable test time pool"
    )
    parser.add_argument(
        "--no-prefetcher",
        action="store_true",
        default=False,
        help="disable fast prefetcher",
    )
    parser.add_argument(
        "--pin-mem",
        action="store_true",
        default=False,
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument(
        "--channels-last",
        action="store_true",
        default=False,
        help="Use channels_last memory layout",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.",
    )
    parser.add_argument(
        "--apex-amp",
        action="store_true",
        default=False,
        help="Use NVIDIA Apex AMP mixed precision",
    )
    parser.add_argument(
        "--native-amp",
        action="store_true",
        default=False,
        help="Use Native Torch AMP mixed precision",
    )
    parser.add_argument(
        "--tf-preprocessing",
        action="store_true",
        default=False,
        help="Use Tensorflow preprocessing pipeline (require CPU TF installed",
    )
    parser.add_argument(
        "--use-ema",
        dest="use_ema",
        action="store_true",
        help="use ema version of weights if present",
    )
    parser.add_argument(
        "--torchscript",
        dest="torchscript",
        action="store_true",
        help="convert model torchscript for inference",
    )
    parser.add_argument(
        "--legacy-jit",
        dest="legacy_jit",
        action="store_true",
        help="use legacy jit mode for pytorch 1.5/1.5.1/1.6 to get back fusion performance",
    )
    parser.add_argument(
        "--results-file",
        default="",
        type=str,
        metavar="FILENAME",
        help="Output csv file for validation results (summary)",
    )
    parser.add_argument(
        "--real-labels",
        default="",
        type=str,
        metavar="FILENAME",
        help="Real labels JSON file for imagenet evaluation",
    )
    parser.add_argument(
        "--valid-labels",
        default="",
        type=str,
        metavar="FILENAME",
        help="Valid label indices txt file for validation of partial label space",
    )
    parser.add_argument(
        "--use-inference-mode",
        dest="use_inference_mode",
        action="store_true",
        default=False,
        help="use inference mode version of model definition.",
    )

    parser.add_argument(
        "--eval-mode",
        default="logits",
        choices=["logits", "zeroshot", "linearprobe"],
        help=(
            "Evaluation mode: 'logits' uses classification head, 'zeroshot' uses CLIP text "
            "embeddings, 'linearprobe' trains a logistic-regression head on frozen projected embeddings."
        ),
    )
    parser.add_argument(
        "--dataset",
        default="",
        type=str,
        help="Dataset identifier for zero-shot prompts (e.g. 'food101'). If empty, falls back to --dataset.",
    )
    parser.add_argument(
        "--zeroshot-backbone",
        default="ViT-L/14",
        type=str,
        help="CLIP backbone to use for zero-shot text embeddings.",
    )
    parser.add_argument(
        "--zeroshot-classes-dir",
        default="",
        type=str,
        help="Optional override path to CLIP classes txt directory.",
    )
    parser.add_argument(
        "--zeroshot-templates-dir",
        default="",
        type=str,
        help="Optional override path to CLIP templates txt directory.",
    )

    parser.add_argument(
        "--linearprobe-C",
        default=0.316,
        type=float,
        help="Inverse regularization strength C for logistic regression.",
    )
    parser.add_argument(
        "--linearprobe-max-iter",
        default=1000,
        type=int,
        help="Maximum iterations for logistic regression optimizer.",
    )

    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text
