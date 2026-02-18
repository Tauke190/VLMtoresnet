"""
Optimized Attention Map Extraction for FastViT.

Uses forward patching (not hooks) to capture attention directly from MHSA modules.

Optimizations:
1. Store attention maps on GPU (not CPU) - avoids synchronization overhead
2. Capture from last block only (network.7.5) - not all 6 blocks - avoids unnecessary computation
3. Patch MHSA forward to capture attention directly - more efficient than recomputing Q,K,V
4. Minimal computational overhead during forward pass
"""

import torch
import torch.nn as nn


class AttentionMapExtractor:
    """
    Extract attention maps from FastViT last block (network.7.5 MHSA layer only).

    Optimized for training efficiency - captures only the final attention map.

    Usage:
        model = fastvit_sa36_lrtokens()
        extractor = AttentionMapExtractor(model)

        output = model(x)
        attn_maps = extractor.attention_maps  # Dict with one entry: last block attention

        extractor.remove_patches()  # Clean up when done
    """

    def __init__(self, model):
        """
        Initialize extractor by patching stage4 MHSA modules.

        Args:
            model: FastViT model instance
        """
        self.model = model
        self.attention_maps = {}
        self.original_forwards = {}
        self._patch_mhsa_modules()

    def _patch_mhsa_modules(self):
        """Patch MHSA forward to capture attention from last block only (network.7.5)."""
        registered_count = 0
        last_mhsa_name = None

        # First pass: find the last MHSA module in stage4
        for name, module in self.model.named_modules():
            if module.__class__.__name__ == 'MHSA' and 'network.7' in name:
                last_mhsa_name = name

        # Second pass: patch only the last MHSA module
        if last_mhsa_name:
            for name, module in self.model.named_modules():
                if name == last_mhsa_name:
                    self._patch_single_mhsa(module, name)
                    registered_count += 1
                    break

        if registered_count == 0:
            print("[WARNING] No stage4 MHSA modules found! Check model architecture.")

    def _patch_single_mhsa(self, module, name):
        """
        Patch a single MHSA module's forward method.

        Replaces forward to capture attention before dropout + projection.
        """
        original_forward = module.forward
        extractor = self

        def patched_forward(x):
            """Forward that captures attention on GPU (before dropout)."""
            shape = x.shape
            B, C, H, W = shape
            N = H * W

            # Reshape input
            if len(shape) == 4:
                x_flat = torch.flatten(x, start_dim=2).transpose(-2, -1)
            else:
                x_flat = x

            # Compute QKV
            qkv = (
                module.qkv(x_flat)
                .reshape(B, N, 3, module.num_heads, module.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv.unbind(0)

            # Compute and capture attention (on GPU, before dropout)
            attn = (q * module.scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)

            # CAPTURE HERE: Store on GPU to avoid sync overhead
            extractor.attention_maps[name] = attn.detach()

            # Continue with original forward (dropout + projection)
            attn = module.attn_drop(attn)
            x_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x_out = module.proj(x_out)
            x_out = module.proj_drop(x_out)

            if len(shape) == 4:
                x_out = x_out.transpose(-2, -1).reshape(B, C, H, W)

            return x_out

        # Store original and apply patch
        self.original_forwards[name] = original_forward
        module.forward = patched_forward

    def __call__(self, x):
        """
        Run model forward pass and capture attention maps.

        Args:
            x: Input tensor [batch_size, 3, height, width]

        Returns:
            model_output: Output from the model
        """
        self.attention_maps.clear()

        with torch.no_grad():
            output = self.model(x)

        return output, self.attention_maps

    def remove_patches(self):
        """Restore original MHSA forward methods. Call when done."""
        for name, original_forward in self.original_forwards.items():
            # Find module and restore
            for module_name, module in self.model.named_modules():
                if module_name == name:
                    module.forward = original_forward
                    break
        self.original_forwards.clear()

    def get_layer_attention(self, layer_name):
        """
        Get attention maps for a specific layer.

        Args:
            layer_name: Name of the layer

        Returns:
            Attention tensor of shape [batch_size, num_heads, num_tokens, num_tokens] on GPU
        """
        return self.attention_maps.get(layer_name, None)

    def list_attention_layers(self):
        """List all attention layers being captured (last block of stage4 only)."""
        layers = []
        last_mhsa_name = None

        # Find the last MHSA module in stage4
        for name, module in self.model.named_modules():
            if module.__class__.__name__ == 'MHSA' and 'network.7' in name:
                last_mhsa_name = name

        if last_mhsa_name:
            layers.append(last_mhsa_name)

        return layers