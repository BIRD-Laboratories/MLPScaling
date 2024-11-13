import argparse
import os
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np
from PIL import Image

# Define the MLP model (same as in the training script)
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Custom Dataset class to handle image preprocessing (same as in the training script)
class TinyImageNetDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        img = example['image']
        img = np.array(img.convert('L'))  # Convert PIL image to grayscale NumPy array
        img = img.reshape(-1)  # Flatten the image
        img = torch.from_numpy(img).float()  # Convert to tensor
        label = torch.tensor(example['label'])
        return img, label

# Function to evaluate the model on the validation set and compute class-wise accuracy
def evaluate_model(model, val_loader, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    class_correct[label.item()] += 1
                class_total[label.item()] += 1

    class_accuracies = {}
    for class_idx in range(num_classes):
        if class_total[class_idx] > 0:
            class_accuracies[class_idx] = 100 * class_correct[class_idx] / class_total[class_idx]
        else:
            class_accuracies[class_idx] = 0.0

    return class_accuracies

# Main function to load the model and evaluate it
def main():
    parser = argparse.ArgumentParser(description='Evaluate the MLP model on the zh-plus/tiny-imagenet dataset.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--layer_count', type=int, default=2, help='Number of hidden layers (default: 2)')
    parser.add_argument('--width', type=int, default=512, help='Number of neurons per hidden layer (default: 512)')
    parser.add_argument('--output_file', type=str, default='class_accuracies.txt', help='Output file to save class-wise accuracies')
    args = parser.parse_args()

    # Load the zh-plus/tiny-imagenet dataset
    dataset = load_dataset('zh-plus/tiny-imagenet')
    val_dataset = dataset['valid']  # Assuming 'validation' is the correct key

    # Determine the number of classes
    num_classes = len(set(val_dataset['label']))

    # Determine the fixed resolution of the images
    image_size = 64  # Assuming the images are square

    # Define the model
    input_size = image_size * image_size  # Since images are grayscale
    hidden_sizes = [args.width] * args.layer_count
    output_size = num_classes

    model = MLP(input_size, hidden_sizes, output_size)
    model.load_state_dict(torch.load(args.checkpoint))

    # Create DataLoader for validation
    val_loader = DataLoader(TinyImageNetDataset(val_dataset), batch_size=8, shuffle=False)

    # Evaluate the model
    class_accuracies = evaluate_model(model, val_loader, num_classes)

    # Print the results
    print("Class-wise accuracies:")
    for class_idx, accuracy in class_accuracies.items():
        print(f"Class {class_idx}: {accuracy:.2f}%")

    # Save the results to a text file
    with open(args.output_file, 'w') as f:
        for class_idx, accuracy in class_accuracies.items():
            f.write(f"Class {class_idx}: {accuracy:.2f}%\n")

if __name__ == '__main__':
    main()
