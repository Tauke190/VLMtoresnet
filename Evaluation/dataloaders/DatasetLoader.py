import os
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from PIL import Image
from ucf101 import UCF101
from datasets import load_dataset


class UniversalDatasetWrapper(Dataset):
    """Wrapper to convert HuggingFace dataset to PyTorch Dataset format."""

    def __init__(self, hf_ds, transform):
        self.ds = hf_ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item["image"]

        # Ensure PIL Image
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if img.mode != "RGB":
            img = img.convert("RGB")

        label = int(item["label"])

        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_stanford_cars_loaders(
    data_root, transform, batch_size=128, num_workers=8, download=False
):
    """
    Stanford Cars (HuggingFace version): 196 classes
    Train: 8,144 | Test: 8,041

    Replaces torchvision StanfordCars loader.
    Uses UniversalDatasetWrapper to match all other datasets.
    """

    # Load HF dataset (automatically cached under ~/.cache/huggingface)
    hf_train = load_dataset("tanganke/stanford_cars", split="train")
    hf_test = load_dataset("tanganke/stanford_cars", split="test")

    # Wrap them to behave like PyTorch datasets
    train_dataset = UniversalDatasetWrapper(hf_train, transform)
    test_dataset = UniversalDatasetWrapper(hf_test, transform)

    # Build loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def get_gtsrb_loaders(
    data_root, transform, batch_size=128, num_workers=8, download=False
):
    """
    GTSRB: 43 classes | Train: 39,209 | Test: 12,630
    """
    root = os.path.join(data_root, "gtsrb")

    train_dataset = datasets.GTSRB(
        root=root, split="train", transform=transform, download=download
    )
    test_dataset = datasets.GTSRB(
        root=root, split="test", transform=transform, download=download
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def get_food101_loaders(
    data_root, transform, batch_size=128, num_workers=8, download=False
):
    """
    Food101: 101 classes | Train: 75,750 | Test: 25,250
    """
    root = os.path.join(data_root, "food101")

    train_dataset = datasets.Food101(
        root=root, split="train", transform=transform, download=download
    )
    test_dataset = datasets.Food101(
        root=root, split="test", transform=transform, download=download
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def get_aircraft_loaders(
    data_root, transform, batch_size=128, num_workers=8, download=False
):
    """
    FGVC Aircraft: 100 classes | Train: 6,667 | Test: 3,333
    """
    root = os.path.join(data_root, "fgvc_aircraft")

    train_dataset = datasets.FGVCAircraft(
        root=root, split="trainval", transform=transform, download=download
    )
    test_dataset = datasets.FGVCAircraft(
        root=root, split="test", transform=transform, download=download
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def get_sst2_loaders(
    data_root, transform, batch_size=128, num_workers=8, download=False
):
    """
    Rendered SST2 (HuggingFace 'nateraw/rendered-sst2'): 2 classes
    Labels: 0 = negative, 1 = positive

    Note: data_root is unused; data is loaded via HuggingFace datasets
    into its cache (~/.cache/huggingface).
    """
    # Load HF dataset
    ds = load_dataset("nateraw/rendered-sst2")
    hf_train = ds["train"]
    hf_val = ds["validation"]  # we’ll treat validation as test for zero-shot

    # Wrap to behave like standard PyTorch datasets
    train_dataset = UniversalDatasetWrapper(hf_train, transform)
    test_dataset = UniversalDatasetWrapper(hf_val, transform)

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def get_fer2013_loaders(
    data_root, transform, batch_size=128, num_workers=8, download=False
):
    """
    FER2013 (torchvision).

    Expects:
        data_root/
            FER2013/
                fer2013/
                    fer2013.csv
                    train/
                    test/
    """
    root = os.path.join(data_root, "FER2013")

    train_dataset = datasets.FER2013(
        root=root,
        split="train",
        transform=transform,
    )
    test_dataset = datasets.FER2013(
        root=root,
        split="test",
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def get_country211_loaders(
    data_root, transform, batch_size=128, num_workers=8, download=True
):
    """
    Country211 (torchvision).

    Expects (auto-created when download=True):
        data_root/
            country211/
                ...
    """
    root = data_root  # torchvision will create data_root/country211/

    train_split = datasets.Country211(
        root=root,
        split="train",
        transform=transform,
        download=download,
    )
    val_split = datasets.Country211(
        root=root,
        split="valid",
        transform=transform,
        download=download,
    )
    train_dataset = ConcatDataset([train_split, val_split])

    test_dataset = datasets.Country211(
        root=root,
        split="test",
        transform=transform,
        download=download,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def get_ucf101_loaders(
    data_root, transform, batch_size=128, num_workers=8, download=False
):
    root = os.path.join(data_root, "ucf101")

    train_dataset = UCF101(root=root, train=True, transform=transform)
    test_dataset = UCF101(root=root, train=False, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
