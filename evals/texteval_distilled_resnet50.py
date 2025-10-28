import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import timm
from tqdm import tqdm
import clip  # Requires OpenAI's CLIP library

# --- Configuration ---
VAL_DIR = '/home/av354855/data/datasets/imagenet/val'  # Path to your validation dataset
VAL_SUBSET_SIZE = 50000
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Official CLIP Prompt Templates ---
PROMPT_TEMPLATES = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]

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

            # Ensure image_features is in float32 to match text_features
            image_features = image_features.float()

            # Compute similarity between image features and text features
            logits = image_features @ text_features.T  # Shape: [batch_size, num_classes]
            predicted = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# --- Main Evaluation ---
def run_text_based_evaluation():
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

    # 3. Prepare the validation dataset
    data_config = timm.data.resolve_model_data_config(student)
    val_transform = timm.data.create_transform(**data_config, is_training=False)
    val_dataset = ImageFolder(root=VAL_DIR, transform=val_transform)
    val_subset = Subset(val_dataset, range(min(VAL_SUBSET_SIZE, len(val_dataset))))
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    class_names = val_dataset.classes  # Class names from the dataset

    # 4. Load the text encoder (e.g., CLIP's text encoder)
    print("Loading text encoder...")
    text_model, preprocess = clip.load("ViT-L/14", device=DEVICE)  # Use the same CLIP model as during distillation
    tokenizer = clip.tokenize

    # 5. Generate text features
    print("Generating text features...")
    text_features = get_text_features(class_names, text_model, tokenizer)

    # 6. Run validation
    print("Starting text-based evaluation...")
    accuracy = validate_with_text(student, val_loader, text_features, projection_layer)
    print(f"Text-Based Validation Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    run_text_based_evaluation()