import torch
import os

output_dir = "checkpoints/CLIPtoResNet/"  # Change this to your actual output directory
proj_path = os.path.join(output_dir, "projector_best_aircraft.pth.tar")

if os.path.exists(proj_path):
    checkpoint = torch.load(proj_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    print("Keys in projector state_dict:")
    for k in state_dict.keys():
        print(k)
else:
    print(f"Checkpoint not found at {proj_path}")