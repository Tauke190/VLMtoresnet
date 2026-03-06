"""
Debug script to compare feature norms in FastViT_Projector vs FastViT_Adapter
to identify scaling issues in the adapter implementation.
"""

import sys
import torch
import torch.nn as nn
from functools import partial

# Add models to path
from models.fastvit_proposed import FastViT_Projector, FastViT_adapter
from models.fastvit import RepCPE

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model config
fastvit_sa36_config = dict(
    layers=[6, 6, 18, 6],
    embed_dims=[64, 128, 256, 512],
    mlp_ratios=[4, 4, 4, 4],
    downsamples=[True, True, True, True],
    pos_embs=[None, None, None, None],
    token_mixers=("repmixer", "repmixer", "repmixer", "attention"),
    layer_scale_init_value=1e-6,
    num_classes=1000,
    freeze_backbone=True,
    clip_dim=768,
)

def hook_fn(name):
    """Create a hook function to capture feature norms"""
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            norm = output.norm(p=2).item()
            logger.info(f"{name}: norm = {norm:.6f}, shape = {output.shape}")
    return hook

def register_hooks_on_blocks(model, model_name):
    """Register hooks on all block layers to monitor feature norms"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Registering hooks for {model_name}")
    logger.info(f"{'='*60}\n")
    
    handles = []
    
    # Hook on patch embed
    handles.append(
        model.patch_embed.register_forward_hook(hook_fn(f"{model_name}/patch_embed"))
    )
    
    # Hooks on network blocks
    for stage_idx, block in enumerate(model.network):
        if isinstance(block, nn.Sequential):
            for block_idx, sub_block in enumerate(block):
                block_type = type(sub_block).__name__
                handle = sub_block.register_forward_hook(
                    hook_fn(f"{model_name}/stage_{stage_idx}/block_{block_idx}_{block_type}")
                )
                handles.append(handle)
        else:
            # Position embedding or other block
            block_type = type(block).__name__
            handle = block.register_forward_hook(
                hook_fn(f"{model_name}/block_{stage_idx}_{block_type}")
            )
            handles.append(handle)
    
    return handles

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dummy input
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    logger.info(f"Input shape: {x.shape}, norm: {x.norm(p=2).item():.6f}\n")
    
    # ========== Test FastViT_Projector ==========
    logger.info("\n" + "="*80)
    logger.info("TESTING FastViT_Projector")
    logger.info("="*80)
    
    model_projector = FastViT_Projector(**fastvit_sa36_config).to(device)
    model_projector.eval()
    
    handles_proj = register_hooks_on_blocks(model_projector, "Projector")
    
    with torch.no_grad():
        logger.info("Forward pass for Projector...\n")
        proj_embed, proj_cls, proj_feat = model_projector(x)
        logger.info(f"\nProjector output - embedding norm: {proj_embed.norm(p=2).item():.6f}")
        logger.info(f"Projector output - logits norm: {proj_cls.norm(p=2).item():.6f}")
        logger.info(f"Projector output - features norm: {proj_feat.norm(p=2).item():.6f}")
    
    # Remove hooks
    for handle in handles_proj:
        handle.remove()
    
    # ========== Test FastViT_Adapter ==========
    logger.info("\n" + "="*80)
    logger.info("TESTING FastViT_Adapter")
    logger.info("="*80)
    
    model_adapter = FastViT_adapter(**fastvit_sa36_config, adapter_reduction=4).to(device)
    model_adapter.eval()
    
    handles_adapt = register_hooks_on_blocks(model_adapter, "Adapter")
    
    with torch.no_grad():
        logger.info("Forward pass for Adapter...\n")
        adapt_embed, adapt_cls, adapt_feat = model_adapter(x)
        logger.info(f"\nAdapter output - embedding norm: {adapt_embed.norm(p=2).item():.6f}")
        logger.info(f"Adapter output - logits norm: {adapt_cls.norm(p=2).item():.6f}")
        logger.info(f"Adapter output - features norm: {adapt_feat.norm(p=2).item():.6f}")
    
    # Remove hooks
    for handle in handles_adapt:
        handle.remove()
    
    # ========== Detailed Comparison ==========
    logger.info("\n" + "="*80)
    logger.info("COMPARISON")
    logger.info("="*80)
    
    logger.info(f"Projector embedding norm: {proj_embed.norm(p=2).item():.6f}")
    logger.info(f"Adapter embedding norm:   {adapt_embed.norm(p=2).item():.6f}")
    logger.info(f"Ratio (Adapter/Projector): {(adapt_embed.norm(p=2).item() / proj_embed.norm(p=2).item()):.4f}")
    
    logger.info(f"\nProjector features norm: {proj_feat.norm(p=2).item():.6f}")
    logger.info(f"Adapter features norm:   {adapt_feat.norm(p=2).item():.6f}")
    logger.info(f"Ratio (Adapter/Projector): {(adapt_feat.norm(p=2).item() / proj_feat.norm(p=2).item()):.4f}")
    
    # Check layer scales
    logger.info("\n" + "="*80)
    logger.info("LAYER SCALE VALUES")
    logger.info("="*80)
    
    logger.info("\nProjector block layer scales:")
    for stage_idx, block in enumerate(model_projector.network):
        if isinstance(block, nn.Sequential):
            for block_idx, sub_block in enumerate(block):
                block_type = type(sub_block).__name__
                if hasattr(sub_block, 'layer_scale'):
                    scale_val = sub_block.layer_scale.data.mean().item()
                    logger.info(f"  Stage {stage_idx}, Block {block_idx} ({block_type}): {scale_val:.6e}")
                elif hasattr(sub_block, 'layer_scale_1'):
                    scale_val1 = sub_block.layer_scale_1.data.mean().item()
                    scale_val2 = sub_block.layer_scale_2.data.mean().item()
                    logger.info(f"  Stage {stage_idx}, Block {block_idx} ({block_type}): scale_1={scale_val1:.6e}, scale_2={scale_val2:.6e}")
    
    logger.info("\nAdapter block layer scales:")
    for stage_idx, block in enumerate(model_adapter.network):
        if isinstance(block, nn.Sequential):
            for block_idx, sub_block in enumerate(block):
                block_type = type(sub_block).__name__
                if hasattr(sub_block, 'layer_scale'):
                    scale_val = sub_block.layer_scale.data.mean().item()
                    logger.info(f"  Stage {stage_idx}, Block {block_idx} ({block_type}): {scale_val:.6e}")
                elif hasattr(sub_block, 'layer_scale_1'):
                    scale_val1 = sub_block.layer_scale_1.data.mean().item()
                    scale_val2 = sub_block.layer_scale_2.data.mean().item()
                    logger.info(f"  Stage {stage_idx}, Block {block_idx} ({block_type}): scale_1={scale_val1:.6e}, scale_2={scale_val2:.6e}")

if __name__ == "__main__":
    main()
