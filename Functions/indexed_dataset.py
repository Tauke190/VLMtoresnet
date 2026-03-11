"""
Thin wrapper around a dataset that also returns the sample index.

Used by CRD loss which needs to know the dataset position of each sample
to maintain its memory bank.
"""

from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image


class IndexedDatasetWrapper(Dataset):
    """Wraps any dataset to also return the sample index.

    Returns (image, target, index) instead of (image, target).
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data = self.dataset[index]
        # data is typically (image, target) or (image, target, *extra)
        return (*data, index)

    def __len__(self):
        return len(self.dataset)

    def __setattr__(self, name, value):
        # Forward transform sets to inner dataset (timm's create_loader sets dataset.transform)
        if name != 'dataset' and hasattr(self, 'dataset'):
            setattr(self.dataset, name, value)
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        # Delegate attribute access to the wrapped dataset (e.g., .reader, .transform)
        return getattr(self.dataset, name)


def collate_with_indices(batch):
    """Custom collate function that handles PIL images and optional indices.

    Handles batches in the form:
    - (image, target) → (tensor_batch, target_batch)
    - (image, target, index) → (tensor_batch, target_batch, index_batch)

    Converts PIL images to numpy arrays (uint8) and then to tensors,
    matching timm's fast_collate behavior.
    """
    if not batch:
        return None

    # Check if we have indices (3-tuple) or not (2-tuple)
    has_indices = len(batch[0]) >= 3
    batch_size = len(batch)

    # Extract components
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    indices = [item[2] for item in batch] if has_indices else None

    # Convert targets to tensor
    targets_tensor = torch.tensor(targets, dtype=torch.int64)

    # Convert indices to tensor if present
    if indices is not None:
        indices_tensor = torch.tensor(indices, dtype=torch.long)

    # Handle image conversion (PIL → tensor)
    # Check first image type
    first_img = images[0]
    if isinstance(first_img, Image.Image):
        # Convert PIL images to numpy arrays and check sizes
        img_arrays = [np.array(img) for img in images]
        first_shape = img_arrays[0].shape

        # Create tensor with first image's shape
        images_tensor = torch.zeros((batch_size, *first_shape), dtype=torch.uint8)

        for i, img_array in enumerate(img_arrays):
            # Resize image to first image's size if different
            if img_array.shape != first_shape:
                img_pil = Image.fromarray(img_array) if isinstance(img_array, np.ndarray) else img_array
                img_pil = img_pil.resize((first_shape[1], first_shape[0]), Image.BILINEAR)
                img_array = np.array(img_pil)
            images_tensor[i] = torch.from_numpy(img_array)
    elif isinstance(first_img, np.ndarray):
        # Convert numpy arrays to tensor
        first_shape = first_img.shape
        images_tensor = torch.zeros((batch_size, *first_shape), dtype=torch.uint8)
        for i, img in enumerate(images):
            if img.shape != first_shape:
                # Resize if needed
                img_pil = Image.fromarray(img)
                img_pil = img_pil.resize((first_shape[1], first_shape[0]), Image.BILINEAR)
                img = np.array(img_pil)
            images_tensor[i] = torch.from_numpy(img)
    elif isinstance(first_img, torch.Tensor):
        # When prefetcher is disabled, timm applies transforms (ToTensor + Normalize) before
        # collate, so images arrive as float32 CHW tensors already normalized. Just stack them.
        images_tensor = torch.stack(images)
    else:
        raise TypeError(f"Unsupported image type: {type(first_img)}")

    # Only transpose BHWC→BCHW for PIL/numpy images (uint8 HWC format).
    # Torch tensors from transforms are already in CHW format.
    if images_tensor.dtype == torch.uint8:
        images_tensor = images_tensor.permute(0, 3, 1, 2).contiguous()

    # Return based on whether we have indices
    if has_indices:
        return images_tensor, targets_tensor, indices_tensor
    else:
        return images_tensor, targets_tensor