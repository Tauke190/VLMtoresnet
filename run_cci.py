#!/usr/bin/env python
"""
CCI (Cluster-based Concept Importance) Visualization Script

Run CCI analysis on images to visualize which regions drive CLIP's predictions.

Usage:
    python run_cci.py --image path/to/image.jpg --text "a photo of a dog"
    python run_cci.py --image-dir path/to/images/ --text "description"
    python run_cci.py --image image.jpg --text "prompt1" "prompt2" --compare-prompts
"""

import argparse
import os
import sys
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures

from CCI_visualize import CCIVisualizer


def parse_args():
    parser = argparse.ArgumentParser(
        description='CCI (Cluster-based Concept Importance) Visualization for CLIP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image analysis
  python run_cci.py --image dog.jpg --text "a photo of a dog"

  # Batch processing
  python run_cci.py --image-dir ./images/ --text "an animal"

  # Compare multiple prompts
  python run_cci.py --image cat.jpg --text "a cat" "a pet" "an animal" --compare-prompts

  # Use different CLIP model
  python run_cci.py --image dog.jpg --text "a dog" --model ViT-L/14
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--image', type=str,
        help='Path to a single image file'
    )
    input_group.add_argument(
        '--image-dir', type=str,
        help='Path to directory containing images'
    )

    # Text prompt
    parser.add_argument(
        '--text', type=str, nargs='+', required=True,
        help='Text prompt(s) for CCI analysis'
    )

    # Model settings
    parser.add_argument(
        '--model', type=str, default='ViT-B/32',
        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'],
        help='CLIP model variant (default: ViT-B/32)'
    )
    parser.add_argument(
        '--n-clusters', type=int, default=7,
        help='Number of clusters for K-means (default: 7)'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device to use (default: auto-detect cuda/cpu)'
    )

    # Output options
    parser.add_argument(
        '--output-dir', type=str, default='./cci_outputs',
        help='Directory to save output visualizations (default: ./cci_outputs)'
    )
    parser.add_argument(
        '--save-detailed', action='store_true',
        help='Save detailed cluster-by-cluster visualization'
    )
    parser.add_argument(
        '--compare-prompts', action='store_true',
        help='Compare CCI results across multiple text prompts'
    )
    parser.add_argument(
        '--no-show', action='store_true',
        help='Do not display figures (only save)'
    )

    # Visualization options
    parser.add_argument(
        '--interpolation', type=str, default='bicubic',
        choices=['bicubic', 'spline', 'gaussian', 'lanczos'],
        help='Heatmap interpolation method (default: bicubic)'
    )
    parser.add_argument(
        '--colormap', type=str, default='jet',
        help='Colormap for heatmap (jet, hot, viridis, inferno, etc. default: jet)'
    )
    parser.add_argument(
        '--alpha', type=float, default=0.5,
        help='Overlay transparency 0-1 (default: 0.5)'
    )

    # Image processing
    parser.add_argument(
        '--extensions', type=str, nargs='+',
        default=['.jpg', '.jpeg', '.png', '.bmp', '.webp'],
        help='Image file extensions to process (default: .jpg .jpeg .png .bmp .webp)'
    )

    return parser.parse_args()


def get_image_files(image_dir: str, extensions: list) -> list:
    """Get all image files in directory with specified extensions."""
    image_dir = Path(image_dir)
    files = []
    for ext in extensions:
        files.extend(image_dir.glob(f'*{ext}'))
        files.extend(image_dir.glob(f'*{ext.upper()}'))
    return sorted(files)


def process_single_image(
    cci: CCIVisualizer,
    image_path: str,
    text_prompts: list,
    output_dir: str,
    save_detailed: bool = False,
    compare_prompts: bool = False,
    show: bool = True,
    interpolation: str = 'bicubic',
    colormap: str = 'jet',
    alpha: float = 0.5,
):
    """Process a single image with CCI analysis."""
    print(f"\nProcessing: {image_path}")

    # Load image
    image = Image.open(image_path).convert('RGB')
    image_name = Path(image_path).stem

    if compare_prompts and len(text_prompts) > 1:
        # Compare multiple prompts for the same image
        import matplotlib.pyplot as plt

        n_prompts = len(text_prompts)
        fig, axes = plt.subplots(n_prompts, 4, figsize=(16, 4 * n_prompts))

        if n_prompts == 1:
            axes = [axes]

        for idx, text in enumerate(text_prompts):
            results = cci.compute_cci(image, text)

            # Original image
            display_img = image.resize((cci.input_resolution, cci.input_resolution))
            axes[idx][0].imshow(display_img)
            axes[idx][0].set_title(f'Original\nSim: {results["original_similarity"][0]:.3f}')
            axes[idx][0].axis('off')

            # Clusters
            cluster_vis = results['cluster_labels'][0].reshape(cci.grid_size, cci.grid_size)
            axes[idx][1].imshow(cluster_vis, cmap='tab10')
            axes[idx][1].set_title(f'Clusters (K={cci.n_clusters})')
            axes[idx][1].axis('off')

            # Smooth importance map
            importance_map = results['importance_map'][0]
            smooth_heatmap = cci.generate_smooth_heatmap(
                importance_map,
                target_size=(cci.input_resolution, cci.input_resolution),
                method=interpolation
            )
            im = axes[idx][2].imshow(smooth_heatmap, cmap=colormap, interpolation='bilinear')
            axes[idx][2].set_title('Importance Map')
            axes[idx][2].axis('off')

            # Smooth overlay
            overlay = cci.create_heatmap_overlay(
                display_img, importance_map,
                colormap=colormap, alpha=alpha, method=interpolation
            )
            axes[idx][3].imshow(overlay)
            axes[idx][3].set_title(f'Text: "{text}"')
            axes[idx][3].axis('off')

        plt.tight_layout()
        save_path = os.path.join(output_dir, f'{image_name}_compare.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved comparison: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    else:
        # Single prompt or process each prompt separately
        for text in text_prompts:
            safe_text = text.replace(' ', '_').replace('/', '-')[:30]

            # Basic visualization with smooth heatmap
            save_path = os.path.join(output_dir, f'{image_name}_{safe_text}.png')
            results = cci.visualize(
                image, text,
                save_path=save_path,
                show=show,
                interpolation=interpolation,
                colormap=colormap,
                alpha=alpha,
            )
            print(f"  Text: '{text}'")
            print(f"    Similarity: {results['original_similarity'][0]:.4f}")
            print(f"    Top cluster importance: {results['importance_scores'][0].max():.4f}")

            # Detailed visualization
            if save_detailed:
                detail_path = os.path.join(output_dir, f'{image_name}_{safe_text}_detail.png')
                cci.visualize_clusters_detail(
                    image, text,
                    save_path=detail_path,
                    show=show,
                    interpolation=interpolation,
                )


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("CCI (Cluster-based Concept Importance) Visualization")
    print("=" * 60)

    # Initialize CCI
    print(f"\nInitializing CLIP model: {args.model}")
    print(f"Number of clusters: {args.n_clusters}")

    cci = CCIVisualizer(
        model_name=args.model,
        device=args.device,
        n_clusters=args.n_clusters,
    )

    print(f"Device: {cci.device}")
    print(f"Input resolution: {cci.input_resolution}x{cci.input_resolution}")
    print(f"Grid size: {cci.grid_size}x{cci.grid_size} ({cci.n_patches} patches)")

    # Get image files
    if args.image:
        image_files = [args.image]
    else:
        image_files = get_image_files(args.image_dir, args.extensions)
        print(f"\nFound {len(image_files)} images in {args.image_dir}")

    if not image_files:
        print("No images found to process!")
        sys.exit(1)

    # Process images
    print(f"\nText prompt(s): {args.text}")
    print(f"Interpolation: {args.interpolation}, Colormap: {args.colormap}")
    print(f"Output directory: {args.output_dir}")
    print("-" * 60)

    for image_path in image_files:
        try:
            process_single_image(
                cci=cci,
                image_path=str(image_path),
                text_prompts=args.text,
                output_dir=args.output_dir,
                save_detailed=args.save_detailed,
                compare_prompts=args.compare_prompts,
                show=not args.no_show,
                interpolation=args.interpolation,
                colormap=args.colormap,
                alpha=args.alpha,
            )
        except Exception as e:
            print(f"  Error processing {image_path}: {e}")
            continue

    print("\n" + "=" * 60)
    print(f"Done! Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
