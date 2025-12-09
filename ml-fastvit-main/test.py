import torch
from timm.models import create_model
import clip

device = "cuda"

# FastViT
model = create_model("fastvit_sa36", pretrained=False).to(device)
model.eval()

x = torch.randn(2, 3, 256, 256, device=device)
with torch.no_grad():
    if hasattr(model, "forward_features"):
        feats = model.forward_features(x)
    else:
        feats = model(x)
    if isinstance(feats, (tuple, list)):
        feats = feats[0]
    print("FastViT final feature map shape:", feats.shape)

# CLIP
clip_model, _ = clip.load("ViT-L/14", device=device, jit=False)
with torch.no_grad():
    tokens = clip.tokenize(["a dog"]).to(device)
    txt = clip_model.encode_text(tokens)
    print("CLIP text feature shape:", txt.shape)