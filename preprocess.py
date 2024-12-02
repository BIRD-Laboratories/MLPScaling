import argparse
import torch
from PIL import Image
import numpy as np
import json
from datasets import Dataset
from transformers import AutoModel
from sklearn.model_selection import train_test_split

# Function to preprocess images
def preprocess_image(image, size=(224, 224)):
    image = image.convert('RGB').resize(size)
    img_array = np.array(image) / 255.0
    img_tensor = torch.tensor((img_array - 0.5) * 2, dtype=torch.float32).permute(2, 0, 1)
    return img_tensor

# Main function
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Encode images from a local dataset.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the local JSON dataset.')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the HuggingFace encoder model.')
    args = parser.parse_args()

    # Load the dataset from the JSON file
    with open(args.dataset_path, 'r') as f:
        data = json.load(f)
    
    # Create a Dataset from the list of dictionaries
    dataset = Dataset.from_list(data)

    # Initialize the encoder model and set to evaluation mode
    encoder = AutoModel.from_pretrained(args.model_name)
    encoder.eval()

    # Move encoder to CUDA device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device)

    # Initialize list to hold the data
    encoded_data = []

    # Set batch size
    batch_size = 32

    # Process the dataset in batches
    for i in range(0, len(dataset), batch_size):
        # Get a batch of image paths and labels
        batch_image_paths = [dataset[j]['image_path'] for j in range(i, i+batch_size) if j < len(dataset)]
        batch_labels = [dataset[j]['label'] for j in range(i, i+batch_size) if j < len(dataset)]
        
        # Load and preprocess images
        batch_images = []
        for path in batch_image_paths:
            try:
                img = Image.open(path)
                batch_images.append(preprocess_image(img))
            except Exception as e:
                print(f"Error loading image at {path}: {e}")
                continue
        
        if not batch_images:
            continue  # Skip empty batches
        
        # Stack into a batch tensor and move to device
        batch_tensor = torch.stack(batch_images, dim=0).to(device)
        
        # Encode the batch
        with torch.no_grad():
            outputs = encoder(batch_tensor)
            encoded = outputs.pooler_output  # Adjust based on the encoder's output structure
        
        # Convert encoded features to lists
        encoded_features = encoded.cpu().numpy().tolist()
        
        # Collect data
        for feat, label in zip(encoded_features, batch_labels):
            encoded_data.append({'codebook': feat, 'label': label})
        
        # Print progress
        print(f'Processed {i + len(batch_image_paths)} / {len(dataset)} images')

    # Save the encoded data to a JSON file
    with open('dataset/encoded_tiny_imagenet.json', 'w') as f:
        json.dump(encoded_data, f)

if __name__ == '__main__':
    main()

