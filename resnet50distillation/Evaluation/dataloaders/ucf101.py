import os
import json
from PIL import Image
from torch.utils.data import Dataset


class UCF101(Dataset):
    """
    UCF101 Mid-Frames Dataset (Zhou split)
    """

    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        self.frames_dir = os.path.join(root, "UCF-101-midframes")
        json_path = os.path.join(root, "split_zhou_UCF101.json")

        # Load JSON
        with open(json_path, "r") as f:
            obj = json.load(f)

        if train:
            split = obj["train"] + obj["val"]
        else:
            split = obj["test"]

        # Build absolute paths and labels
        self.images = [os.path.join(self.frames_dir, i[0]) for i in split]
        self.labels = [i[1] for i in split]  # label_id already provided

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label
