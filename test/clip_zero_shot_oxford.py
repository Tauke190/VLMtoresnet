import os
import torch
import open_clip # Using open_clip is still fine
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

def evaluate_clip_with_local_dataset():
    """
    Evaluates a pre-trained CLIP model on a local dataset
    using the open_clip and torchvision libraries.
    """
    # 1. Setup: Load Model and Define Device
    # =======================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Load a standard CLIP model using open_clip
    # ==================================================
    # CHANGE: Use 'ViT-B-32' and pretrained='openai'. This is a standard
    # model available in all versions of the library.
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32',         # Model architecture
        pretrained='openai',  # Use the original OpenAI pre-trained weights
        device=device
    )
    # Get the tokenizer for the model
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    print("CLIP model (ViT-B/32) and preprocessor loaded successfully.")
    
    # 3. Dataset: Load Local Data with ImageFolder
    # ============================================
    dataset_path = os.path.expanduser('~/data/datasets/oxford_pet/val')
    print(f"Loading dataset from: {dataset_path}")

    try:
        test_dataset = ImageFolder(root=dataset_path, transform=preprocess)
    except FileNotFoundError:
        print(f"Error: Dataset not found at the specified path: {dataset_path}")
        return

    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    class_names = test_dataset.classes
    print(f"Found {len(class_names)} classes (breeds).")

    # 4. Create and Tokenize Text Prompts
    # ===================================
    text_prompts = [f"a photo of a {c.replace('_', ' ')}" for c in class_names]
    print("Example prompt:", text_prompts[0])

    text_inputs = tokenizer(text_prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # 5. Evaluation Loop
    # ==================
    all_predictions = []
    all_labels = []

    print("\nStarting evaluation...")
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        predictions = similarity.argmax(dim=-1)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # 6. Calculate and Print Final Accuracy
    # =====================================
    if len(all_labels) == 0:
        print("\nNo images were found in the dataset. Cannot calculate accuracy.")
        return
        
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels)) * 100
    
    print("\n----- Evaluation Complete -----")
    print(f"Total Images Evaluated: {len(all_labels)}")
    print(f"Correct Predictions: {np.sum(np.array(all_predictions) == np.array(all_labels))}")
    print(f"Zero-Shot Accuracy: {accuracy:.2f}%")
    print("-----------------------------")

if __name__ == "__main__":
    evaluate_clip_with_local_dataset()