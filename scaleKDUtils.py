import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNtoViTTokenDistiller(nn.Module):
    def __init__(
        self,
        cnn_feature_dim,
        teacher_token_dim,
        teacher_num_tokens,
        patch_size=2,
        transformer_width=768,
        num_heads=16,
        num_layers=1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.teacher_token_dim = teacher_token_dim
        self.teacher_num_tokens = teacher_num_tokens

        # Project CNN channels to transformer width if needed
        self.channel_proj = nn.Linear(
            cnn_feature_dim * patch_size * patch_size, transformer_width
        ) if cnn_feature_dim * patch_size * patch_size != transformer_width else nn.Identity()

        # Positional embedding for all patches
        self.pos_embed = None  # Will be initialized after seeing input

        # Trainable queries (same as teacher tokens)
        self.trainable_queries = nn.Parameter(
            torch.randn(1, teacher_num_tokens, transformer_width)
        )

        # Transformer decoder block
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=transformer_width, nhead=num_heads, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # Project to teacher token dim if needed
        self.token_proj = nn.Linear(
            transformer_width, teacher_token_dim
        ) if transformer_width != teacher_token_dim else nn.Identity()

    def patchify(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Feature map size must be divisible by patch size"
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # [B, C, nH, nW, pH, pW]
        x = x.permute(0,2,3,1,4,5).contiguous()
        nH, nW = x.shape[1], x.shape[2]
        x = x.view(B, nH * nW, C * self.patch_size * self.patch_size)
        print(f"Patchified shape: {x.shape} (B, num_patches, patch_dim)")
        return x

    def forward(self, cnn_feat):
        # cnn_feat: [B, C, H, W]
        print(f"Input CNN feature shape: {cnn_feat.shape}")
        x = self.patchify(cnn_feat)  # [B, num_patches, patch_dim]
        x = self.channel_proj(x)     # [B, num_patches, transformer_width]
        print(f"After channel_proj: {x.shape}")

        # Positional embedding
        B, N, D = x.shape
        if (self.pos_embed is None) or (self.pos_embed.shape[1] != N):
            self.pos_embed = nn.Parameter(torch.zeros(1, N, D, device=x.device))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        x = x + self.pos_embed
        print(f"After adding pos_embed: {x.shape}")

        # Expand trainable queries for batch
        queries = self.trainable_queries.expand(B, -1, -1)  # [B, num_queries, D]
        print(f"Trainable queries shape: {queries.shape}")

        # Transformer decoder: queries attend to patch tokens
        out = self.transformer_decoder(tgt=queries, memory=x)  # [B, num_queries, D]
        print(f"After transformer decoder: {out.shape}")

        out = self.token_proj(out)  # [B, num_queries, teacher_token_dim]
        print(f"Output queries after projection: {out.shape}")

        return out

# Example usage in your training loop:
# 1. Get pre-pooling feature map from ResNet-50 (e.g., after layer4, before avgpool)
# 2. Pass through CNNtoViTTokenDistiller
# 3. Compute MSE loss with teacher tokens

# Example patch for ResNet-50 backbone to expose pre-pooling features:
def get_resnet50_prepool_features(model, x):
    # model: timm resnet50, x: [B,3,224,224]
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.act1(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    # x: [B, 2048, 7, 7]
    return x

# In your training loop, replace student_features = get_student_features(backbone, images)
# with:
# cnn_feat = get_resnet50_prepool_features(backbone, images)
# student_tokens = cnn_to_vit_token_distiller(cnn_feat)
# teacher_tokens = ... # get from teacher (e.g., teacher.visual(images))
# If needed, match teacher_tokens shape to [B, num_tokens, teacher_token_dim]
# loss = F.mse_loss(student_tokens, teacher_tokens)