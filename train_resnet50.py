import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
import time
from math import sqrt
import sys

## Import classes from other files
from models.resnet50 import ResNet50

# Import PyTorch libraries
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# TRAIN_DIR = '/home/c3-0/datasets/ImageNet/train'
# VAL_DIR = '/home/c3-0/datasets/ImageNet/validation'

TRAIN_DIR = '~/data/datasets/imagenet/train'
VAL_DIR = '~/data/datasets/imagenet/val'

EPOCHS = 50

## Set Hyperparameters
class Params:
    def __init__(self):
        self.batch_size = 32
        self.name = "resnet_50_sgd1"
        self.workers = 4
        self.lr = 0.1
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

        if batch % 100 == 0:
            batch_loss = running_loss / total_samples if total_samples > 0 else 0.0
            batch_acc = 100.0 * running_correct / total_samples if total_samples > 0 else 0.0
            logger.info(f"Epoch {epoch+1} Batch {batch}: loss={batch_loss:.6f}, acc={batch_acc:.2f}%")
            sys.stdout.flush()

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_acc = 100.0 * running_correct / total_samples if total_samples > 0 else 0.0
    epoch_time = time.time() - start0
    logger.info(f"Epoch {epoch+1} completed: loss={epoch_loss:.6f}, acc={epoch_acc:.2f}%, time={epoch_time:.2f}s")
    sys.stdout.flush()


def test(dataloader, model, loss_fn, epoch, writer, train_dataloader, calc_acc5=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct, correct_top5 = 0, 0, 0

    # Use tqdm for progress visualization
    progress_bar = tqdm(dataloader, desc=f"Testing Epoch {epoch+1}", file=sys.stdout, dynamic_ncols=False, ascii=True)

    with torch.no_grad():
        for X, y in progress_bar:
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
    # TensorBoard disabled:
    # if writer is not None:
    #     writer.add_scalar('test loss', test_loss, step)
    #     writer.add_scalar('test accuracy', accuracy, step)
    #     if calc_acc5:
    #         writer.add_scalar('test accuracy5', top5_accuracy, step)

    logger.info(f"Test Results - Epoch {epoch+1}: Accuracy={accuracy:.2f}%, Avg loss={test_loss:.6f}")
    sys.stdout.flush()
    if calc_acc5:
        logger.info(f"Top-5 Accuracy={top5_accuracy:.2f}%")
        sys.stdout.flush()

if __name__ == "__main__":
    params = Params()
    print(params, params.batch_size)

    training_folder_name = TRAIN_DIR
    val_folder_name = VAL_DIR

    train_transformation = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BILINEAR),  # remove antialias for compatibility
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
        ])

    train_dataset = torchvision.datasets.ImageFolder(
        root=training_folder_name,
        transform=train_transformation
    )
    train_sampler = torch.utils.data.RandomSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        sampler=train_sampler,
        num_workers = params.workers,
        pin_memory=True,
    )

    val_transformation = transforms.Compose([
            transforms.Resize(size=256),  # avoid antialias kw for wider compatibility
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
        ])
    val_dataset = torchvision.datasets.ImageFolder(
        root=val_folder_name,
        transform=val_transformation
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=32,
        num_workers=params.workers,
        shuffle=False,
        pin_memory=True
    )

    # device
    print("Libraries imported - ready to use PyTorch", torch.__version__)
    sys.stdout.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using {device} device")
    sys.stdout.flush()

    ## Testing with pre-trained model : only to be done once
    ## testing a pretrained model to validate correctness of our dataset, transform and metrics code
    # pretrained_model = torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT').to(device)
    # start = time.time()
    # loss_fn = nn.CrossEntropyLoss()
    # test(val_loader, pretrained_model, loss_fn, epoch=0, writer=None, train_dataloader=train_loader, calc_acc5=True)
    # print("Elapsed: ", time.time() - start)

    # resume training options
    resume_training = True

    num_classes = len(train_dataset.classes)
    model = ResNet50(num_classes=num_classes)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=params.lr, momentum=params.momentum, weight_decay=params.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.lr_step_size, gamma=params.lr_gamma)

    ## Current State of Training
    start_epoch = 0
    checkpoint_path = os.path.join("checkpoints", params.name, f"checkpoint.pth")
    print(checkpoint_path)
    sys.stdout.flush()

    if resume_training and os.path.exists(checkpoint_path):
        print("Resuming training from checkpoint")
        print(checkpoint_path)
        sys.stdout.flush()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint["epoch"] + 1
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        assert params == checkpoint["params"]

    # from torch.utils.tensorboard import SummaryWriter
    from pathlib import Path
    Path(os.path.join("checkpoints", params.name)).mkdir(parents=True, exist_ok=True)
    # writer = SummaryWriter('runs/' + params.name)
    writer = None  # TensorBoard disabled
    test(val_loader, model, loss_fn, epoch=0, writer=writer, train_dataloader=train_loader, calc_acc5=True)
    print("Starting training")
    sys.stdout.flush()
    for epoch in range(start_epoch, EPOCHS):
        print(f"Epoch {epoch}")
        sys.stdout.flush()
        train(train_loader, model, loss_fn, optimizer, epoch=epoch, writer=writer)
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "params": params
        }
        torch.save(checkpoint, os.path.join("checkpoints", params.name, f"model_{epoch}.pth"))
        torch.save(checkpoint, os.path.join("checkpoints", params.name, f"checkpoint.pth"))
        lr_scheduler.step()
        test(val_loader, model, loss_fn, epoch + 1, writer, train_dataloader=train_loader, calc_acc5=True)
