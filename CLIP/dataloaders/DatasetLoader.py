import os
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from PIL import Image

try:
    from ucf101 import UCF101
except ImportError:
    UCF101 = None

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


class HFDatasetWrapper(Dataset):
    """Wrapper for HuggingFace datasets."""

    def __init__(self, hf_ds, transform):
        self.ds = hf_ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        label = int(item["label"])
        if self.transform:
            img = self.transform(img)
        return img, label


def make_loaders(train_ds, test_ds, batch_size, num_workers):
    """Create dataloaders."""
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def get_stanford_cars_loaders(
    data_root, transform, batch_size=128, num_workers=8, download=False
):
    cache_dir = os.path.join(data_root, "stanford_cars")
    hf_train = load_dataset(
        "tanganke/stanford_cars", split="train", cache_dir=cache_dir
    )
    hf_test = load_dataset("tanganke/stanford_cars", split="test", cache_dir=cache_dir)
    train_ds = HFDatasetWrapper(hf_train, transform)
    test_ds = HFDatasetWrapper(hf_test, transform)
    return make_loaders(train_ds, test_ds, batch_size, num_workers)


def get_gtsrb_loaders(
    data_root, transform, batch_size=128, num_workers=8, download=False
):
    root = os.path.join(data_root, "gtsrb")
    train_ds = datasets.GTSRB(
        root=root, split="train", transform=transform, download=download
    )
    test_ds = datasets.GTSRB(
        root=root, split="test", transform=transform, download=download
    )
    return make_loaders(train_ds, test_ds, batch_size, num_workers)


def get_food101_loaders(
    data_root, transform, batch_size=128, num_workers=8, download=False
):
    root = os.path.join(data_root, "food101")
    train_ds = datasets.Food101(
        root=root, split="train", transform=transform, download=download
    )
    test_ds = datasets.Food101(
        root=root, split="test", transform=transform, download=download
    )
    return make_loaders(train_ds, test_ds, batch_size, num_workers)


def get_aircraft_loaders(
    data_root, transform, batch_size=128, num_workers=8, download=False
):
    root = os.path.join(data_root, "fgvc_aircraft")
    train_ds = datasets.FGVCAircraft(
        root=root, split="trainval", transform=transform, download=download
    )
    test_ds = datasets.FGVCAircraft(
        root=root, split="test", transform=transform, download=download
    )
    return make_loaders(train_ds, test_ds, batch_size, num_workers)


def get_sst2_loaders(
    data_root, transform, batch_size=128, num_workers=8, download=False
):
    ds = load_dataset("nateraw/rendered-sst2")
    train_ds = HFDatasetWrapper(ds["train"], transform)
    test_ds = HFDatasetWrapper(ds["validation"], transform)
    return make_loaders(train_ds, test_ds, batch_size, num_workers)


def get_fer2013_loaders(
    data_root, transform, batch_size=128, num_workers=8, download=False
):
    root = os.path.join(data_root, "FER2013")
    train_ds = datasets.FER2013(root=root, split="train", transform=transform)
    test_ds = datasets.FER2013(root=root, split="test", transform=transform)
    return make_loaders(train_ds, test_ds, batch_size, num_workers)


def get_country211_loaders(
    data_root, transform, batch_size=128, num_workers=8, download=True
):
    train_split = datasets.Country211(
        root=data_root, split="train", transform=transform, download=download
    )
    val_split = datasets.Country211(
        root=data_root, split="valid", transform=transform, download=download
    )
    train_ds = ConcatDataset([train_split, val_split])
    test_ds = datasets.Country211(
        root=data_root, split="test", transform=transform, download=download
    )
    return make_loaders(train_ds, test_ds, batch_size, num_workers)


def get_ucf101_loaders(
    data_root, transform, batch_size=128, num_workers=8, download=False
):
    root = os.path.join(data_root, "ucf101")
    train_ds = UCF101(root=root, train=True, transform=transform)
    test_ds = UCF101(root=root, train=False, transform=transform)
    return make_loaders(train_ds, test_ds, batch_size, num_workers)


DATASET_LOADERS = {
    "stanford_cars": get_stanford_cars_loaders,
    "gtsrb": get_gtsrb_loaders,
    "food101": get_food101_loaders,
    "aircraft": get_aircraft_loaders,
    "sst2": get_sst2_loaders,
    "fer2013": get_fer2013_loaders,
    "country211": get_country211_loaders,
    "ucf101": get_ucf101_loaders,
}
