


from timm.models import create_model
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from CLIP.dataloaders import Food101


import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.fastvit import fastvit_sa36

from timm.models import create_model

# wget https://docs-assets.developer.apple.com/ml-research/models/fastvit/image_classification_models/fastvit_sa36.pth.tar
# mv fastvit_sa36_reparam.pth.tar Weights/
# https://docs-assets.developer.apple.com/ml-research/models/fastvit/image_classification_distilled_models/fastvit_sa36_reparam.pth.tar

def get_features(dataset, mode='backbone1'):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=200)):
            if mode == 'backbone1' :
                features = model.forward_backbone(images.to(device))
                B, C, H, W  = features.shape 
                features = features.reshape(B, C, -1)
                features = features.mean(-1)
            elif mode == 'backbone2' :
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
model.eval()      
device = 'cuda'
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
MODE='backbone2'
MODE='classifier'
# Calculate the image features
train_features, train_labels = get_features(train, mode=MODE)
test_features, test_labels = get_features(test, mode=MODE)
# Perform logistic regression
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)

# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
print(f"Accuracy  :: {MODE} = {accuracy:.3f}")


# python linear_probing_demo.py

# MODE='backbone1'
# Accuracy = 72.131

# MODE='backbone2'
# Accuracy = 

# MODE='classifier'
# Accuracy = 

# MODE='classifier'
# Accuracy = 