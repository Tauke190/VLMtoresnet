import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import timm
from tqdm import tqdm

# --- Configuration ---
VAL_DIR = '/home/av354855/data/datasets/imagenet/val'  # Path to your validation dataset
TRAIN_DIR = '/home/av354855/data/datasets/imagenet/train'  # Path to your training dataset
VAL_SUBSET_SIZE = 50000
TRAIN_SUBSET_SIZE = 50000
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Utility Functions ---
def get_student_features(model, images, projection_layer):
    """Extract global average pooled features from the student (ResNet-50) and project to 768 dimensions."""
    feature_map = model.forward_features(images)
    pooled_features = model.global_pool(feature_map)  # Shape: [batch_size, 2048]
    projected_features = projection_layer(pooled_features)  # Shape: [batch_size, 768]
    return projected_features

def extract_features(model, dataloader, projection_layer):
    """Extract features from the model for all images in the dataloader."""
    model.eval()
    projection_layer.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(DEVICE)
            features = get_student_features(model, images, projection_layer)
            features = features / features.norm(dim=-1, keepdim=True)  # Normalize
            all_features.append(features.cpu())
            all_labels.append(labels)
    return torch.cat(all_features), torch.cat(all_labels)

def knn_evaluation(train_features, train_labels, val_features, val_labels, k=5):
    """Perform k-NN evaluation."""
    correct = 0
    total = val_features.size(0)

    for i in tqdm(range(total), desc="k-NN Evaluation"):
        # Compute cosine similarity between the validation feature and all training features
        similarities = torch.mm(train_features, val_features[i].unsqueeze(1)).squeeze(1)

        # Get the indices of the top-k most similar training features
        _, indices = similarities.topk(k)

        # Get the labels of the top-k neighbors
        neighbor_labels = train_labels[indices]

        # Predict the majority label
        predicted_label = torch.mode(neighbor_labels).values.item()

        # Check if the prediction is correct
        if predicted_label == val_labels[i].item():
            correct += 1

    accuracy = 100 * correct / total
    return accuracy

# --- Main Evaluation ---
def run_knn_evaluation():
    print(f"Using device: {DEVICE}")

    # 1. Load the student backbone (distilled)
    print("Loading distilled student backbone...")
    student = timm.create_model('resnet50', pretrained=False, num_classes=0).to(DEVICE)
    student.load_state_dict(torch.load('../resnet50_distilled_backbone.pth', map_location=DEVICE))
    student.eval()

    # 2. Load the projection layer
    print("Loading projection layer...")
    projection_layer = nn.Linear(2048, 768).to(DEVICE)

    # Load the projection layer weights (768 → 2048)
    distillation_weights = torch.load('../projection_layer.pth', map_location=DEVICE)

    # Transpose the weights for evaluation (2048 → 768)
    evaluation_weights = {
        'weight': distillation_weights['weight'].T,  # Transpose the weight matrix
        'bias': distillation_weights['bias'][:768]  # Slice the bias to match the output dimension
    }

    # Load the transposed weights into the evaluation projection layer
    projection_layer.load_state_dict(evaluation_weights)
    projection_layer.eval()

    # 3. Prepare the training and validation datasets
    data_config = timm.data.resolve_model_data_config(student)
    transform = timm.data.create_transform(**data_config, is_training=False)

    train_dataset = ImageFolder(root=TRAIN_DIR, transform=transform)
    val_dataset = ImageFolder(root=VAL_DIR, transform=transform)

    train_subset = Subset(train_dataset, range(min(TRAIN_SUBSET_SIZE, len(train_dataset))))
    val_subset = Subset(val_dataset, range(min(VAL_SUBSET_SIZE, len(val_dataset))))

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 4. Extract features
    print("Extracting training features...")
    train_features, train_labels = extract_features(student, train_loader, projection_layer)

    print("Extracting validation features...")
    val_features, val_labels = extract_features(student, val_loader, projection_layer)

    # 5. Perform k-NN evaluation
    print("Starting k-NN evaluation...")
    k = 5  # You can adjust k as needed
    accuracy = knn_evaluation(train_features, train_labels, val_features, val_labels, k=k)
    print(f"k-NN Validation Accuracy (k={k}): {accuracy:.2f}%")

if __name__ == "__main__":
    run_knn_evaluation()