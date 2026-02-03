"""
Cluster-based Concept Importance (CCI) Visualization for CLIP

Implementation based on the paper:
"Concept Regions Matter: Benchmarking CLIP with a New Cluster-Importance Approach"
(OpenReview: https://openreview.net/forum?id=K7wkjqLjrt)

CCI is a training-free interpretability method that:
1. Extracts patch embeddings from CLIP's Vision Transformer
2. Clusters patches using K-means into semantically coherent regions
3. Masks each cluster by setting attention logits to -inf
4. Measures similarity drop to compute cluster importance
5. Generates spatial importance maps for visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline
from typing import List, Tuple, Optional, Union
import os
import sys

# Add CLIP to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CLIP'))
from clip import clip


class CCIVisualizer:
    """
    Cluster-based Concept Importance (CCI) Visualizer for CLIP models.

    CCI identifies which image regions drive CLIP's predictions by:
    - Grouping patches into semantic clusters
    - Masking clusters and measuring prediction changes
    - Creating interpretable importance heatmaps
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = None,
        n_clusters: int = 7,
    ):
        """
        Initialize the CCI Visualizer.

        Args:
            model_name: CLIP model variant (e.g., "ViT-B/32", "ViT-B/16", "ViT-L/14")
            device: Device to run on (auto-detected if None)
            n_clusters: Number of clusters for K-means (default: 7 as in paper)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_clusters = n_clusters

        # Load CLIP model
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        # Get model configuration
        self.visual = self.model.visual
        if not hasattr(self.visual, 'transformer'):
            raise ValueError("CCI requires a Vision Transformer model (ViT), not ResNet")

        self.patch_size = self.visual.conv1.kernel_size[0]
        self.input_resolution = self.visual.input_resolution
        self.grid_size = self.input_resolution // self.patch_size
        self.n_patches = self.grid_size ** 2
        self.embed_dim = self.visual.conv1.out_channels

        # Store hooks for extracting intermediate features
        self._patch_embeddings = None
        self._hooks = []

    def _register_hooks(self):
        """Register forward hooks to extract patch embeddings."""
        def hook_fn(module, input, output):
            # Output shape: [L, N, D] where L = num_patches + 1 (CLS), N = batch, D = embed_dim
            self._patch_embeddings = output.permute(1, 0, 2)  # [N, L, D]

        # Hook after ln_pre to get initial patch embeddings
        handle = self.visual.ln_pre.register_forward_hook(hook_fn)
        self._hooks.append(handle)

    def _remove_hooks(self):
        """Remove registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks = []

    def extract_patch_embeddings(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract patch embeddings from the Vision Transformer.

        Args:
            image: Preprocessed image tensor [B, C, H, W]

        Returns:
            Patch embeddings [B, N, D] where N = grid_size^2 (excluding CLS token)
        """
        with torch.no_grad():
            # Get patch embeddings through conv1
            x = self.visual.conv1(image.type(self.model.dtype))  # [B, D, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, D, N]
            x = x.permute(0, 2, 1)  # [B, N, D]

            # Add positional embeddings (excluding CLS position)
            pos_embed = self.visual.positional_embedding[1:].to(x.dtype)  # [N, D]
            x = x + pos_embed

            # Apply layer norm
            x = self.visual.ln_pre(x)

        return x

    def cluster_patches(
        self,
        patch_embeddings: torch.Tensor,
        n_clusters: int = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster patch embeddings using K-means.

        Args:
            patch_embeddings: Patch embeddings [B, N, D]
            n_clusters: Number of clusters (uses default if None)

        Returns:
            cluster_labels: Cluster assignment for each patch [B, N]
            cluster_centers: Cluster centers [B, K, D]
        """
        n_clusters = n_clusters or self.n_clusters
        batch_size = patch_embeddings.shape[0]

        # Convert to numpy for sklearn
        embeddings_np = patch_embeddings.cpu().float().numpy()

        all_labels = []
        all_centers = []

        for b in range(batch_size):
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
            )
            labels = kmeans.fit_predict(embeddings_np[b])
            all_labels.append(labels)
            all_centers.append(kmeans.cluster_centers_)

        return np.array(all_labels), np.array(all_centers)

    def _encode_image_with_mask(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode image with attention masking for specified patches.

        This implements the key CCI mechanism: setting attention logits to -inf
        for masked patches prevents the CLS token from aggregating their information.

        Args:
            image: Preprocessed image tensor [B, C, H, W]
            mask: Binary mask [B, N] where 1 indicates patches to mask

        Returns:
            Image features [B, D]
        """
        batch_size = image.shape[0]

        # Create attention mask: shape [L, L] where L = N + 1 (patches + CLS)
        # We want to prevent CLS (position 0) from attending to masked patches
        L = self.n_patches + 1

        # Start with no masking
        attn_mask = torch.zeros(L, L, device=self.device, dtype=image.dtype)

        # For each masked patch, set the CLS->patch attention to -inf
        # mask shape: [B, N], we need to handle batch dimension
        # Since nn.MultiheadAttention expects [L, L] mask (same for all batch),
        # we'll process one sample at a time or use a custom forward

        # Build custom forward pass with masking
        with torch.no_grad():
            # Patch embedding
            x = self.visual.conv1(image.type(self.model.dtype))
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)  # [B, N, D]

            # Add CLS token
            cls_token = self.visual.class_embedding.to(x.dtype)
            cls_tokens = cls_token + torch.zeros(batch_size, 1, x.shape[-1],
                                                  dtype=x.dtype, device=x.device)
            x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, D]

            # Add positional embeddings
            x = x + self.visual.positional_embedding.to(x.dtype)
            x = self.visual.ln_pre(x)

            # Permute for transformer: [L, B, D]
            x = x.permute(1, 0, 2)

            # Create attention mask for all layers
            # Shape: [B, L, L] -> we'll expand mask for multi-head attention
            # mask: [B, N] -> expand to [B, 1, L] for broadcasting
            expanded_mask = torch.zeros(batch_size, L, device=self.device, dtype=x.dtype)
            expanded_mask[:, 1:] = mask.to(x.dtype)  # Skip CLS position (index 0)

            # Convert to attention mask format: -inf for masked positions
            # Shape for additive mask: [B * num_heads, L, L] or [L, L] for same mask
            attn_mask = expanded_mask.unsqueeze(1).expand(-1, L, -1)  # [B, L, L]
            attn_mask = attn_mask * float('-inf')
            attn_mask = torch.nan_to_num(attn_mask, nan=0.0)  # Replace nan with 0

            # Process through transformer blocks with masking
            for block in self.visual.transformer.resblocks:
                x = self._forward_block_with_mask(block, x, attn_mask)

            # Permute back: [B, L, D]
            x = x.permute(1, 0, 2)

            # Extract CLS token and project
            x = self.visual.ln_post(x[:, 0, :])
            if self.visual.proj is not None:
                x = x @ self.visual.proj

        return x

    def _forward_block_with_mask(
        self,
        block: nn.Module,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through a transformer block with custom attention mask.

        Args:
            block: ResidualAttentionBlock
            x: Input tensor [L, B, D]
            attn_mask: Attention mask [B, L, L]

        Returns:
            Output tensor [L, B, D]
        """
        L, B, D = x.shape

        # Self-attention with masking
        ln_x = block.ln_1(x)

        # Manual attention computation to apply mask
        q = F.linear(ln_x, block.attn.in_proj_weight[:D], block.attn.in_proj_bias[:D])
        k = F.linear(ln_x, block.attn.in_proj_weight[D:2*D], block.attn.in_proj_bias[D:2*D])
        v = F.linear(ln_x, block.attn.in_proj_weight[2*D:], block.attn.in_proj_bias[2*D:])

        num_heads = block.attn.num_heads
        head_dim = D // num_heads

        # Reshape for multi-head attention
        q = q.contiguous().view(L, B * num_heads, head_dim).transpose(0, 1)  # [B*H, L, head_dim]
        k = k.contiguous().view(L, B * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(L, B * num_heads, head_dim).transpose(0, 1)

        # Compute attention scores
        scale = head_dim ** -0.5
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * scale  # [B*H, L, L]

        # Apply attention mask (expand for heads)
        if attn_mask is not None:
            # Expand mask: [B, L, L] -> [B*H, L, L]
            expanded_attn_mask = attn_mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
            expanded_attn_mask = expanded_attn_mask.reshape(B * num_heads, L, L)
            attn_weights = attn_weights + expanded_attn_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.bmm(attn_weights, v)  # [B*H, L, head_dim]

        # Reshape back
        attn_output = attn_output.transpose(0, 1).contiguous().view(L, B, D)

        # Output projection
        attn_output = F.linear(attn_output, block.attn.out_proj.weight, block.attn.out_proj.bias)

        # Residual connection
        x = x + attn_output

        # MLP
        x = x + block.mlp(block.ln_2(x))

        return x

    def compute_importance_scores(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        cluster_labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute importance scores for each cluster using the CCI method.

        Args:
            image: Preprocessed image tensor [B, C, H, W]
            text: Tokenized text tensor [B, context_length]
            cluster_labels: Cluster assignments [B, N]

        Returns:
            importance_scores: Normalized importance for each cluster [B, K]
            similarity_drops: Raw similarity drops [B, K]
        """
        batch_size = image.shape[0]
        n_clusters = cluster_labels.max() + 1

        with torch.no_grad():
            # Get original image and text features
            original_image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)

            # Normalize
            original_image_features = F.normalize(original_image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            # Original similarity
            original_similarity = (original_image_features * text_features).sum(dim=-1)  # [B]

            similarity_drops = []

            for k in range(n_clusters):
                # Create mask for cluster k
                mask = torch.tensor(cluster_labels == k, device=self.device, dtype=torch.float32)

                # Encode with mask
                masked_features = self._encode_image_with_mask(image, mask)
                masked_features = F.normalize(masked_features, dim=-1)

                # Compute similarity with masked features
                masked_similarity = (masked_features * text_features).sum(dim=-1)

                # Similarity drop
                drop = original_similarity - masked_similarity
                similarity_drops.append(drop.cpu().numpy())

            similarity_drops = np.stack(similarity_drops, axis=-1)  # [B, K]

            # Normalize importance scores
            # Handle negative drops by clipping to 0
            positive_drops = np.maximum(similarity_drops, 0)
            total_drops = positive_drops.sum(axis=-1, keepdims=True)
            total_drops = np.maximum(total_drops, 1e-8)  # Avoid division by zero
            importance_scores = positive_drops / total_drops

        return importance_scores, similarity_drops

    def generate_importance_map(
        self,
        importance_scores: np.ndarray,
        cluster_labels: np.ndarray,
    ) -> np.ndarray:
        """
        Generate spatial importance map from cluster importance scores.

        Args:
            importance_scores: Importance for each cluster [B, K]
            cluster_labels: Cluster assignments [B, N]

        Returns:
            Spatial importance map [B, H, W] at patch resolution
        """
        batch_size = importance_scores.shape[0]

        importance_maps = []
        for b in range(batch_size):
            # Map cluster importance to each patch
            patch_importance = importance_scores[b, cluster_labels[b]]  # [N]

            # Reshape to spatial grid
            spatial_map = patch_importance.reshape(self.grid_size, self.grid_size)
            importance_maps.append(spatial_map)

        return np.stack(importance_maps, axis=0)

    def generate_smooth_heatmap(
        self,
        importance_map: np.ndarray,
        target_size: Tuple[int, int] = None,
        method: str = 'bicubic',
        sigma: float = None,
    ) -> np.ndarray:
        """
        Generate a smooth, high-resolution heatmap from the patch-level importance map.

        Args:
            importance_map: Importance map at patch resolution [B, grid_H, grid_W] or [grid_H, grid_W]
            target_size: Target output size (H, W). Default is input_resolution.
            method: Interpolation method - 'bicubic', 'spline', 'gaussian', or 'lanczos'
            sigma: Gaussian smoothing sigma (auto-computed if None). Only used with gaussian method.

        Returns:
            Smooth heatmap at target resolution [B, H, W] or [H, W]
        """
        target_size = target_size or (self.input_resolution, self.input_resolution)

        # Handle single map (no batch dimension)
        single_input = importance_map.ndim == 2
        if single_input:
            importance_map = importance_map[np.newaxis, ...]

        batch_size = importance_map.shape[0]
        grid_h, grid_w = importance_map.shape[1], importance_map.shape[2]
        target_h, target_w = target_size

        smooth_maps = []

        for b in range(batch_size):
            # Convert to float32 for scipy compatibility (float16 not supported)
            heatmap = importance_map[b].astype(np.float32)

            if method == 'bicubic':
                # Bicubic interpolation using scipy zoom
                zoom_factors = (target_h / grid_h, target_w / grid_w)
                smooth = ndimage.zoom(heatmap, zoom_factors, order=3, mode='nearest')

            elif method == 'spline':
                # B-spline interpolation for very smooth results
                x = np.linspace(0, 1, grid_w)
                y = np.linspace(0, 1, grid_h)
                spline = RectBivariateSpline(y, x, heatmap, kx=3, ky=3)

                x_new = np.linspace(0, 1, target_w)
                y_new = np.linspace(0, 1, target_h)
                smooth = spline(y_new, x_new)

            elif method == 'gaussian':
                # First upscale with nearest neighbor, then apply gaussian smoothing
                zoom_factors = (target_h / grid_h, target_w / grid_w)
                upscaled = ndimage.zoom(heatmap, zoom_factors, order=0, mode='nearest')

                # Auto-compute sigma based on patch size for smooth blending
                if sigma is None:
                    sigma = self.patch_size / 2.5

                smooth = ndimage.gaussian_filter(upscaled, sigma=sigma)

            elif method == 'lanczos':
                # Use PIL's high-quality Lanczos resampling
                heatmap_uint8 = (heatmap * 255).astype(np.uint8)
                pil_img = Image.fromarray(heatmap_uint8, mode='L')
                pil_resized = pil_img.resize((target_w, target_h), Image.LANCZOS)
                smooth = np.array(pil_resized) / 255.0

            else:
                raise ValueError(f"Unknown interpolation method: {method}. "
                               f"Use 'bicubic', 'spline', 'gaussian', or 'lanczos'")

            # Normalize to [0, 1]
            smooth = np.clip(smooth, 0, None)
            if smooth.max() > 0:
                smooth = smooth / smooth.max()

            smooth_maps.append(smooth)

        result = np.stack(smooth_maps, axis=0)

        if single_input:
            return result[0]
        return result

    def create_heatmap_overlay(
        self,
        image: Union[Image.Image, np.ndarray],
        heatmap: np.ndarray,
        colormap: str = 'jet',
        alpha: float = 0.5,
        method: str = 'bicubic',
    ) -> np.ndarray:
        """
        Create a smooth heatmap overlay on the original image.

        Args:
            image: Original image (PIL Image or numpy array)
            heatmap: Importance heatmap at patch resolution [grid_H, grid_W]
            colormap: Matplotlib colormap name ('jet', 'hot', 'viridis', etc.)
            alpha: Overlay transparency (0-1)
            method: Interpolation method for upscaling

        Returns:
            Blended image with heatmap overlay as numpy array [H, W, 3]
        """
        # Convert image to numpy if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image.convert('RGB'))
        else:
            image_np = image.copy()
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)

        target_size = (image_np.shape[0], image_np.shape[1])

        # Generate smooth heatmap
        smooth_heatmap = self.generate_smooth_heatmap(
            heatmap, target_size=target_size, method=method
        )

        # Apply colormap
        cmap = plt.cm.get_cmap(colormap)
        heatmap_colored = cmap(smooth_heatmap)[:, :, :3]  # Remove alpha channel
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

        # Blend with original image
        blended = (1 - alpha) * image_np + alpha * heatmap_colored
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        return blended

    def compute_cci(
        self,
        image: Union[Image.Image, torch.Tensor],
        text: Union[str, List[str]],
        n_clusters: int = None,
    ) -> dict:
        """
        Compute full CCI analysis for an image-text pair.

        Args:
            image: PIL Image or preprocessed tensor
            text: Text prompt(s)
            n_clusters: Number of clusters (uses default if None)

        Returns:
            Dictionary containing:
                - patch_embeddings: Extracted patch embeddings
                - cluster_labels: Cluster assignments
                - importance_scores: Normalized importance per cluster
                - similarity_drops: Raw similarity drops per cluster
                - importance_map: Spatial importance map
                - original_similarity: Original image-text similarity
        """
        n_clusters = n_clusters or self.n_clusters

        # Preprocess image if needed
        if isinstance(image, Image.Image):
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        else:
            image_tensor = image.to(self.device)

        # Tokenize text if needed
        if isinstance(text, str):
            text = [text]
        text_tensor = clip.tokenize(text).to(self.device)

        # Expand text to match batch size if needed
        if text_tensor.shape[0] == 1 and image_tensor.shape[0] > 1:
            text_tensor = text_tensor.expand(image_tensor.shape[0], -1)

        # Step 1: Extract patch embeddings
        patch_embeddings = self.extract_patch_embeddings(image_tensor)

        # Step 2: Cluster patches
        cluster_labels, cluster_centers = self.cluster_patches(patch_embeddings, n_clusters)

        # Step 3: Compute importance scores
        importance_scores, similarity_drops = self.compute_importance_scores(
            image_tensor, text_tensor, cluster_labels
        )

        # Step 4: Generate spatial importance map
        importance_map = self.generate_importance_map(importance_scores, cluster_labels)

        # Compute original similarity for reference
        with torch.no_grad():
            image_features = F.normalize(self.model.encode_image(image_tensor), dim=-1)
            text_features = F.normalize(self.model.encode_text(text_tensor), dim=-1)
            original_similarity = (image_features * text_features).sum(dim=-1).cpu().numpy()

        return {
            'patch_embeddings': patch_embeddings.cpu().numpy(),
            'cluster_labels': cluster_labels,
            'cluster_centers': cluster_centers,
            'importance_scores': importance_scores,
            'similarity_drops': similarity_drops,
            'importance_map': importance_map,
            'original_similarity': original_similarity,
        }

    def visualize(
        self,
        image: Union[Image.Image, torch.Tensor],
        text: str,
        n_clusters: int = None,
        save_path: str = None,
        show: bool = True,
        figsize: Tuple[int, int] = (16, 4),
        interpolation: str = 'bicubic',
        colormap: str = 'jet',
        alpha: float = 0.5,
    ) -> dict:
        """
        Compute CCI and visualize results with smooth heatmaps.

        Args:
            image: PIL Image or preprocessed tensor
            text: Text prompt
            n_clusters: Number of clusters
            save_path: Path to save figure (optional)
            show: Whether to display the figure
            figsize: Figure size
            interpolation: Heatmap interpolation method ('bicubic', 'spline', 'gaussian', 'lanczos')
            colormap: Colormap for heatmap ('jet', 'hot', 'viridis', 'inferno', etc.)
            alpha: Overlay transparency (0-1)

        Returns:
            CCI results dictionary
        """
        # Compute CCI
        results = self.compute_cci(image, text, n_clusters)

        # Get original image for display
        if isinstance(image, Image.Image):
            display_image = image.resize((self.input_resolution, self.input_resolution))
        else:
            # Convert tensor to PIL
            img_np = image[0].cpu().numpy()
            # Denormalize
            mean = np.array([0.48145466, 0.4578275, 0.40821073])
            std = np.array([0.26862954, 0.26130258, 0.27577711])
            img_np = img_np.transpose(1, 2, 0) * std + mean
            img_np = np.clip(img_np, 0, 1)
            display_image = Image.fromarray((img_np * 255).astype(np.uint8))

        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=figsize)

        # 1. Original image
        axes[0].imshow(display_image)
        axes[0].set_title(f'Original Image\nSimilarity: {results["original_similarity"][0]:.3f}')
        axes[0].axis('off')

        # 2. Cluster visualization with smooth boundaries
        cluster_vis = results['cluster_labels'][0].reshape(self.grid_size, self.grid_size)
        # Upscale clusters for display (nearest neighbor to preserve boundaries)
        cluster_upscaled = ndimage.zoom(cluster_vis.astype(float),
                                        self.patch_size, order=0)
        axes[1].imshow(cluster_upscaled, cmap='tab10', interpolation='nearest')
        axes[1].set_title(f'Patch Clusters (K={n_clusters or self.n_clusters})')
        axes[1].axis('off')

        # 3. Smooth importance heatmap
        importance_map = results['importance_map'][0]
        smooth_heatmap = self.generate_smooth_heatmap(
            importance_map,
            target_size=(self.input_resolution, self.input_resolution),
            method=interpolation
        )
        im = axes[2].imshow(smooth_heatmap, cmap=colormap, interpolation='bilinear')
        axes[2].set_title(f'CCI Importance Map\n({interpolation} interpolation)')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        # 4. Smooth overlay
        overlay = self.create_heatmap_overlay(
            display_image, importance_map,
            colormap=colormap, alpha=alpha, method=interpolation
        )
        axes[3].imshow(overlay)
        axes[3].set_title(f'Overlay\nText: "{text}"')
        axes[3].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        # Store smooth heatmap in results
        results['smooth_heatmap'] = smooth_heatmap

        return results

    def visualize_clusters_detail(
        self,
        image: Union[Image.Image, torch.Tensor],
        text: str,
        n_clusters: int = None,
        save_path: str = None,
        show: bool = True,
        interpolation: str = 'gaussian',
    ) -> dict:
        """
        Visualize each cluster's contribution in detail with smooth masks.

        Args:
            image: PIL Image or preprocessed tensor
            text: Text prompt
            n_clusters: Number of clusters
            save_path: Path to save figure
            show: Whether to display
            interpolation: Mask interpolation method ('gaussian', 'bicubic', 'spline')

        Returns:
            CCI results dictionary
        """
        n_clusters = n_clusters or self.n_clusters
        results = self.compute_cci(image, text, n_clusters)

        # Get original image for display
        if isinstance(image, Image.Image):
            display_image = np.array(image.resize((self.input_resolution, self.input_resolution)))
        else:
            img_np = image[0].cpu().numpy()
            mean = np.array([0.48145466, 0.4578275, 0.40821073])
            std = np.array([0.26862954, 0.26130258, 0.27577711])
            img_np = img_np.transpose(1, 2, 0) * std + mean
            img_np = np.clip(img_np, 0, 1)
            display_image = (img_np * 255).astype(np.uint8)

        # Create subplot for each cluster + original + bar chart
        n_cols = min(4, n_clusters + 2)
        n_rows = (n_clusters + 2 + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

        # Original image
        axes[0].imshow(display_image)
        axes[0].set_title(f'Original\nSim: {results["original_similarity"][0]:.3f}')
        axes[0].axis('off')

        # Each cluster with smooth masks
        cluster_labels_2d = results['cluster_labels'][0].reshape(self.grid_size, self.grid_size)

        for k in range(n_clusters):
            ax = axes[k + 1]

            # Create mask and apply smooth interpolation
            mask = (cluster_labels_2d == k).astype(float)
            mask_smooth = self.generate_smooth_heatmap(
                mask,
                target_size=(self.input_resolution, self.input_resolution),
                method=interpolation
            )

            ax.imshow(display_image)
            ax.imshow(mask_smooth, cmap='Reds', alpha=0.6, interpolation='bilinear')

            importance = results['importance_scores'][0, k]
            drop = results['similarity_drops'][0, k]
            ax.set_title(f'Cluster {k}\nImp: {importance:.3f}, Drop: {drop:.3f}')
            ax.axis('off')

        # Bar chart of importance scores
        ax = axes[n_clusters + 1]
        clusters = list(range(n_clusters))
        importance = results['importance_scores'][0]
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        ax.bar(clusters, importance, color=colors)
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Importance')
        ax.set_title('Cluster Importance')
        ax.set_xticks(clusters)

        # Hide unused axes
        for i in range(n_clusters + 2, len(axes)):
            axes[i].axis('off')

        plt.suptitle(f'CCI Analysis: "{text}"', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved detailed visualization to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return results
