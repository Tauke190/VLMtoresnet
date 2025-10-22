# eva_forward_pass.py
#
# Description:
# This script demonstrates how to load a pre-trained EVA-02 transformer model
# from the timm library, process a local image file, and classify it
# to predict the most likely ImageNet class.
#
# Dependencies:
# - torch: The core deep learning framework.
# - timm: Ross Wightman's library of PyTorch image models.
# - Pillow: For image processing.
#
# To install the required libraries, run the following command:
# pip install torch timm Pillow

import torch
import timm
from PIL import Image
import json

def run_eva_classification():
    """
    Loads an EVA-02 model, downloads and processes a sample image,
    and performs a classification.
    """
    # 1. Specify the model name for the newer EVA-02 model.
    model_name = 'eva02_large_patch14_448.mim_in22k_ft_in22k_in1k'

    # 2. Load the pre-trained model from timm.
    print(f"Loading pre-trained model: {model_name}...")
    try:
        model = timm.create_model(model_name, pretrained=True)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have an internet connection and the model name is correct.")
        return

    # 3. Get the data configuration and create the appropriate transforms.
    # timm's create_transform function will set up the correct resizing,
    # cropping, and normalization for the model.
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # 4. Load and preprocess the local image.
    # !!! IMPORTANT !!!
    # Replace 'sample_image.jpg' with the path to your own image file.
    image_path = 'bird.JPEG'
    try:
        img = Image.open(image_path).convert("RGB")
        print(f"\nSuccessfully loaded image from {image_path}")
    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'.")
        print("Please make sure the image file exists and the path is correct.")
        return
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Apply the transformations to the image
    input_tensor = transforms(img).unsqueeze(0) # Add batch dimension
    print(f"Preprocessed image into a tensor with shape: {input_tensor.shape}")

    # 5. Load local ImageNet class labels
    try:
        # Assumes 'imagenet_class_index.json' is in the same directory as the script.
        with open('imagenet_class_index.json', 'r') as f:
            imagenet_labels = json.load(f)
        print("Successfully loaded local ImageNet class labels.")
    except FileNotFoundError:
        print("Error: 'imagenet_class_index.json' not found. Please make sure the file is in the same directory.")
        return
    except json.JSONDecodeError:
        print("Error parsing class labels from 'imagenet_class_index.json'.")
        return


    # 6. Perform the forward pass (inference).
    print("\nRunning classification...")
    with torch.no_grad():
        try:
            output = model(input_tensor)
            print("Inference completed successfully.")
        except Exception as e:
            print(f"Error during inference: {e}")
            return

    # 7. Process the output to get probabilities and top predictions.
    # The output from the model are raw logits. We apply a softmax function
    # to convert them into probabilities.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get the top 5 predictions
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    print("\n--- Top 5 Predictions ---")
    for i in range(top5_prob.size(0)):
        # Convert tensor index to string to use as a key in the JSON dictionary
        class_idx = str(top5_catid[i].item())
        # The typical format is {"0": ["n01440764", "tench"]}, so we get the second element
        class_name = imagenet_labels[class_idx][1]
        confidence = top5_prob[i].item() * 100
        print(f"{i+1}. {class_name}: {confidence:.2f}%")


if __name__ == '__main__':
    run_eva_classification()



