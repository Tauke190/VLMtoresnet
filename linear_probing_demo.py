


from timm.models import create_model
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
from CLIP.dataloaders import Food101


import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.fastvit import fastvit_sa36
from misc.utils import dump_images, save_image
from timm.models import create_model

# wget https://docs-assets.developer.apple.com/ml-research/models/fastvit/image_classification_models/fastvit_sa36.pth.tar
# mv fastvit_sa36_reparam.pth.tar Weights/
# https://docs-assets.developer.apple.com/ml-research/models/fastvit/image_classification_distilled_models/fastvit_sa36_reparam.pth.tar

def get_features(dataset, mode='backbone1'):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=200)):
            dump_images(images, "temp.png")
            if mode == 'backbone1' :
                features = model.forward_backbone(images.to(device))
                B, C, H, W  = features.shape 
                features = features.reshape(B, C, -1)
                features = features.mean(-1)
            elif mode == 'classification_neck' :
                features = model.forward_classification_neck(images.to(device))
            elif mode == 'classifier':
                features = model(images.to(device))

            all_features.append(features.cuda())
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


checkpoint_path='Weights/fastvit_sa36.pth.tar'
model = create_model(
        'fastvit_sa36',
        pretrained=False,
        num_classes=1000,
        checkpoint_path=checkpoint_path,
    )

checkpoint = torch.load('fastvit_sa36.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total params: {total_params:,} ({total_params/1e6:.2f}M)")
print(f"Trainable params: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
      
model.eval()      
device = 'cuda'
num_gpus = torch.cuda.device_count()
print(f"GPUs available: {num_gpus}")
model= model.cuda()

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
)

test = Food101(root="/mnt/SSD2/food-101", train=False, transform=transform)
train = Food101(root="/mnt/SSD2/food-101", train=True, transform=transform)        





MODE='backbone1'
#MODE='classification_neck'
#MODE='classifier'
# Calculate the image features
train_features, train_labels = get_features(train, mode=MODE)
test_features, test_labels = get_features(test, mode=MODE)
# Perform logistic regression
start = time.time()
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)

# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
print(f"Accuracy  :: {MODE} = {accuracy:.3f}")
end = time.time()
print(f"Took {end - start:.4f} seconds")

peak = torch.cuda.max_memory_allocated() / (1024**2)
props = torch.cuda.get_device_properties(device)
total = props.total_memory / (1024**2)
print(f"Peak allocated: {peak:.2f} MB")
print(f"Total GPU memory: {total:.2f} MB")

# python linear_probing_demo.py

# MODE='backbone1'
# Accuracy  :: backbone1 = 72.131

# MODE='classification_neck'
# Accuracy  :: classification_neck = 76.725

# MODE='classifier'
# Accuracy  :: classifier = 74.749

