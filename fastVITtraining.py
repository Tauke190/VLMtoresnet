import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
import time
from math import sqrt
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from fastvit import fastvit_t8

TRAIN_DIR = '/home/c3-0/datasets/ImageNet/train'
VAL_DIR = '/home/c3-0/datasets/ImageNet/validation'

# TRAIN_DIR = '~/data/datasets/imagenet/train'
# VAL_DIR = '~/data/datasets/imagenet/val'

EPOCHS = 300

## Set Hyperparameters
class Params:
    def __init__(self):
        self.batch_size = 128
        self.name = "fastVIT_training"
        self.workers = 8
        self.lr = 0.001
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.lr_step_size = 30
        self.lr_gamma = 0.1

    def __repr__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


#Updating with verbose tqdm train and test functions
from tqdm import tqdm  # For Jupyter-specific progress bar
import logging
import time

# Configure logging for Jupyter
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

def train(dataloader, model, loss_fn, optimizer, epoch, writer):
    size = len(dataloader.dataset)
    model.train()
    start0 = time.time()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    batch_times = []
    batch_start = time.time()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item() * X.size(0)
        running_correct += (pred.argmax(1) == y).sum().item()
        total_samples += X.size(0)

        batch_end = time.time()
        batch_times.append(batch_end - batch_start)
        batch_start = batch_end

        if batch == 99:
            avg_100 = np.mean(batch_times[:100])
            est_epoch_100 = avg_100 * len(dataloader)
            print(f"Estimated time for one epoch (based on 100 batches): {est_epoch_100:.2f}s ({est_epoch_100/60:.2f}min)")
            print(f"Estimated total training time for {EPOCHS} epochs: {est_epoch_100*EPOCHS/3600:.2f}h")
        if batch == 999:
            avg_1000 = np.mean(batch_times[:1000])
            est_epoch_1000 = avg_1000 * len(dataloader)
            print(f"Estimated time for one epoch (based on 1000 batches): {est_epoch_1000:.2f}s ({est_epoch_1000/60:.2f}min)")
            print(f"Estimated total training time for {EPOCHS} epochs: {est_epoch_1000*EPOCHS/3600:.2f}h")

        if batch % 100 == 0:
            batch_loss = running_loss / total_samples if total_samples > 0 else 0.0
            batch_acc = 100.0 * running_correct / total_samples if total_samples > 0 else 0.0
            logger.info(f"Epoch {epoch+1} Batch {batch}: loss={batch_loss:.6f}")

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_acc = 100.0 * running_correct / total_samples if total_samples > 0 else 0.0
    epoch_time = time.time() - start0
    logger.info(f"Epoch {epoch+1} completed: loss={epoch_loss:.6f}, acc={epoch_acc:.2f}%, time={epoch_time:.2f}s")

def test(dataloader, model, loss_fn, epoch, writer, train_dataloader, calc_acc5=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct, correct_top5 = 0, 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            if calc_acc5:
                k = min(5, pred.size(1))
                _, pred_topk = pred.topk(k, 1, largest=True, sorted=True)
                correct_top5 += pred_topk.eq(y.view(-1, 1)).any(dim=1).sum().item()

    test_loss /= max(1, num_batches)
    accuracy = 100 * correct / size if size > 0 else 0.0
    top5_accuracy = (100 * correct_top5 / size) if (calc_acc5 and size > 0) else None

    step = epoch * len(train_dataloader.dataset)
    logger.info(f"Test Results - Epoch {epoch+1}: Accuracy={accuracy:.2f}%, Avg loss={test_loss:.6f}")
    # sys.stdout.flush()
    if calc_acc5:
        logger.info(f"Top-5 Accuracy={top5_accuracy:.2f}%")
        # sys.stdout.flush()

if __name__ == "__main__":
    params = Params()
    print(params, params.batch_size)

    training_folder_name = TRAIN_DIR
    val_folder_name = VAL_DIR

    # Split controls
    TRAIN_FRACTION = 0.2          # use 20% of the full training set
    VAL_PER_CLASS = 5             # taken from within the 20% subset
    SPLIT_SEED = 42
    SPLIT_SAVE_DIR = os.path.join("splits", params.name)
    os.makedirs(SPLIT_SAVE_DIR, exist_ok=True)

    train_transformation = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BILINEAR),  # remove antialias for compatibility
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
        ])

    val_transformation = transforms.Compose([
            transforms.Resize(size=256),  # avoid antialias kw for wider compatibility
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Build datasets pointing to the same training folder (we'll split indices)
    train_dataset_full = torchvision.datasets.ImageFolder(
        root=training_folder_name,
        transform=train_transformation
    )
    val_view_full = torchvision.datasets.ImageFolder(
        root=training_folder_name,
        transform=val_transformation
    )

    split_train_path = os.path.join(SPLIT_SAVE_DIR, "train_indices.npy")
    split_val_path = os.path.join(SPLIT_SAVE_DIR, "val_indices.npy")

    if os.path.exists(split_train_path) and os.path.exists(split_val_path):
        train_indices = np.load(split_train_path).tolist()
        val_indices = np.load(split_val_path).tolist()
    else:
        # Stratified split: first choose 20% per class; then pick VAL_PER_CLASS per class for validation from within that 20%
        targets = train_dataset_full.targets
        num_classes = len(train_dataset_full.classes)

        cls_to_indices = [[] for _ in range(num_classes)]
        for idx, c in enumerate(targets):
            cls_to_indices[c].append(idx)

        g = torch.Generator().manual_seed(SPLIT_SEED)

        subset_indices = []
        for c in range(num_classes):
            cls_idx = cls_to_indices[c]
            if not cls_idx:
                continue
            perm = torch.randperm(len(cls_idx), generator=g).tolist()
            take = max(VAL_PER_CLASS, int(len(cls_idx) * TRAIN_FRACTION))  # ensure we can draw VAL_PER_CLASS
            take = min(take, len(cls_idx))
            subset_indices.extend([cls_idx[i] for i in perm[:take]])

        # From the 20% subset, carve out a small fixed val set per class
        subset_set = set(subset_indices)
        # Rebuild per-class buckets but limited to subset
        subset_cls_to_indices = [[] for _ in range(num_classes)]
        for idx in subset_indices:
            subset_cls_to_indices[targets[idx]].append(idx)

        val_indices = []
        train_indices = []
        for c in range(num_classes):
            cls_sub = subset_cls_to_indices[c]
            if not cls_sub:
                continue
            perm = torch.randperm(len(cls_sub), generator=g).tolist()
            v_take = min(VAL_PER_CLASS, len(cls_sub))
            v_idxs = [cls_sub[perm[i]] for i in range(v_take)]
            t_idxs = [cls_sub[perm[i]] for i in range(v_take, len(cls_sub))]
            val_indices.extend(v_idxs)
            train_indices.extend(t_idxs)

        np.save(split_train_path, np.array(train_indices, dtype=np.int64))
        np.save(split_val_path, np.array(val_indices, dtype=np.int64))

    # Wrap subsets for transforms and correct lengths
    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_dataset = torch.utils.data.Subset(val_view_full, val_indices)

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.workers,
        pin_memory=True,
        persistent_workers=True if params.workers > 0 else False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=params.workers,
        pin_memory=True,
        persistent_workers=True if params.workers > 0 else False
    )

    print(f"Classes: {len(train_dataset_full.classes)} | "
          f"20% subset size: {len(train_indices) + len(val_indices)} | "
          f"Train: {len(train_indices)} | Val: {len(val_indices)}")

    # device
    print("Libraries imported - ready to use PyTorch", torch.__version__)
    # sys.stdout.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using {device} device")

    # resume training options
    resume_training = True

    num_classes = len(train_dataset_full.classes)  # use full class list
    model = fastvit_t8(num_classes=num_classes)
    model.to(device)

    # Print the final feature dimension of the FastViT model
    dummy = torch.randn(1, 3, 224, 224).to(device)
    model.eval()
    with torch.no_grad():
        x = dummy
        x = model.forward_embeddings(x)
        x = model.forward_tokens(x)
        x = model.conv_exp(x)
        x = model.gap(x)
        x = x.view(x.size(0), -1)
        print("Penultimate feature shape (before classifier):", x.shape)
        print("Penultimate feature dimension:", x.shape[-1])

        # If you want the final output (logits) as well:
        logits = model.head(x)
        print("Final output (logits) shape:", logits.shape)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=params.lr, momentum=params.momentum, weight_decay=params.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.lr_step_size, gamma=params.lr_gamma)

    ## Current State of Training
    start_epoch = 0
    checkpoint_path = os.path.join("checkpoints", params.name, f"checkpoint.pth")
    print(checkpoint_path)

    if resume_training and os.path.exists(checkpoint_path):
        print("Resuming training from checkpoint")
        print(checkpoint_path)
        # sys.stdout.flush()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint["epoch"] + 1
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        # Compare params as dicts for checkpoint compatibility
        if hasattr(checkpoint["params"], "__dict__"):
            params = checkpoint["params"]
        else:
            params.__dict__.update(checkpoint["params"])

    from pathlib import Path
    Path(os.path.join("checkpoints", params.name)).mkdir(parents=True, exist_ok=True)
    # writer = SummaryWriter('runs/' + params.name)
    writer = None  # TensorBoard disabled
    test(val_loader, model, loss_fn, epoch=0, writer=writer, train_dataloader=train_loader, calc_acc5=True)
    print("Starting training")
    # sys.stdout.flush()
    epoch_times = []
    total_train_start = time.time()  # <-- Add this line
    for epoch in range(start_epoch, EPOCHS):
        print(f"Epoch {epoch}")
        # sys.stdout.flush()
        start_time = time.time()
        train(train_loader, model, loss_fn, optimizer, epoch=epoch, writer=writer)
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "params": params
        }
        # Save checkpoints with "fastVIT" prefix
        torch.save(checkpoint, os.path.join("checkpoints", params.name, f"fastVIT_model_{epoch}.pth"))
        torch.save(checkpoint, os.path.join("checkpoints", params.name, f"fastVIT_checkpoint.pth"))
        lr_scheduler.step()
        test(val_loader, model, loss_fn, epoch + 1, writer, train_dataloader=train_loader, calc_acc5=True)
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        avg_epoch_time = np.mean(epoch_times)
        print(f"Avg epoch time: {avg_epoch_time:.2f}s")
    total_train_time = time.time() - total_train_start  # <-- Add this line
    print(f"Total training time: {total_train_time:.2f}s ({total_train_time/60:.2f} min, {total_train_time/3600:.2f} h)")  # <-- And this
