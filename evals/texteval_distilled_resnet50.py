import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import timm
from tqdm import tqdm
import clip
import argparse

# --- Configuration ---
VAL_DIR = '/home/av354855/data/datasets/imagenet/val'  # Path to your validation dataset
VAL_SUBSET_SIZE = 50000
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_prompt_templates(filepath):
    """Load prompt templates from a text file, one per line."""
    with open(filepath, 'r') as f:
        templates = [line.strip() for line in f if line.strip()]
    return templates

# --- Utility Functions ---
def get_student_features(model, images, projection_layer):
    """Extract global average pooled features from the student (ResNet-50) and project to 768 dimensions."""
    feature_map = model.forward_features(images)
    pooled_features = model.global_pool(feature_map)  # Shape: [batch_size, 2048]
    projected_features = projection_layer(pooled_features)  # Shape: [batch_size, 768]
    return projected_features

def get_text_features(class_names, text_model, tokenizer):
    """Generate text features for all class names using prompt templates."""
    with torch.no_grad():
        text_features = []
        for class_name in class_names:
            prompts = [template.format(class_name) for template in PROMPT_TEMPLATES]
            tokenized_prompts = tokenizer(prompts).to(DEVICE)
            text_embeds = text_model.encode_text(tokenized_prompts)
            text_features.append(text_embeds.mean(dim=0))  # Average over templates
        text_features = torch.stack(text_features)  # Shape: [num_classes, text_feature_dim]
        return text_features

def validate_with_text(student_model, val_loader, text_features, projection_layer):
    """Perform text-based evaluation."""
    student_model.eval()
    projection_layer.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            image_features = get_student_features(student_model, images, projection_layer)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # Normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Normalize

            # Ensure both tensors are in the same precision (Half)
            image_features = image_features.half()  # Convert to Half precision

            # Compute similarity between image features and text features
            logits = image_features @ text_features.T  # Shape: [batch_size, num_classes]
            predicted = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def zeroshot_validate_student(student_model, projector, class_names, val_loader, text_model, tokenizer, templates, device=DEVICE):
    """
    Performs zero-shot validation using CLIP text embeddings as class prototypes.
    Uses projector only for student visual features, not for text features.
    Calculates top-1 and top-5 accuracy.
    """
    # Prepare class text prompts using all templates
    prompts = [template.format(name) for name in class_names for template in templates]
    with torch.no_grad():
        text_tokens = tokenizer(prompts).to(device)
        text_features = text_model.encode_text(text_tokens).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Normalize

    # Average the text features for each class
    num_templates = len(templates)
    num_classes = len(class_names)
    text_features = text_features.view(num_classes, num_templates, -1).mean(dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Normalize

    top1_correct = 0
    top5_correct = 0
    total = 0
    student_model.eval()
    projector.eval()
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Zero-shot validating"):
            images, labels = images.to(device), labels.to(device)
            image_features = student_model.forward_features(images)
            image_features = projector(image_features)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # Normalize

            logits = image_features @ text_features.T
            _, top5_preds = logits.topk(5, dim=1)
            total += labels.size(0)
            top1_correct += (top5_preds[:, 0] == labels).sum().item()
            top5_correct += (top5_preds == labels.view(-1, 1)).sum().item()
    top1_accuracy = 100 * top1_correct / total
    top5_accuracy = 100 * top5_correct / total
    return top1_accuracy, top5_accuracy

# --- Main Evaluation ---
def run_text_based_evaluation(checkpoint_path):
    print(f"Using device: {DEVICE}")

    # Hardcoded prompt path
    prompt_path = "../prompt/imagenet1k.txt"
    print(f"Loading prompt templates from {prompt_path} ...")
    templates = load_prompt_templates(prompt_path)
    templates = templates[:2]
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path} ...")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    student = timm.create_model('resnet50', pretrained=False, num_classes=0).to(DEVICE)
    student.load_state_dict(checkpoint['student_state_dict'])
    projector = nn.Linear(2048, 768).to(DEVICE)
    projector.load_state_dict(checkpoint['projector_state_dict'])

    # Prepare validation dataset
    data_config = timm.data.resolve_model_data_config(student)
    val_transform = timm.data.create_transform(**data_config, is_training=False)
    val_dataset = ImageFolder(root=VAL_DIR, transform=val_transform)
    val_subset = Subset(val_dataset, range(min(VAL_SUBSET_SIZE, len(val_dataset))))
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    class_names = val_dataset.classes

    # Load CLIP text encoder
    print("Loading text encoder...")
    text_model, preprocess = clip.load("ViT-L/14", device=DEVICE)
    tokenizer = clip.tokenize

    # Zero-shot validation
    print("Starting zero-shot evaluation...")
    top1_acc, top5_acc = zeroshot_validate_student(
        student, projector, class_names, val_loader, text_model, tokenizer, templates, device=DEVICE
    )
    print(f"Zero-shot Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Zero-shot Top-5 Accuracy: {top5_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot evaluation for distilled ResNet50 student.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint file')
    # Remove the prompt argument
    args = parser.parse_args()
    run_text_based_evaluation(args.checkpoint)