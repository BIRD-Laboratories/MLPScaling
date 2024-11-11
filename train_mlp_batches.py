import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset

# Define the MLP model
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

# Custom Dataset class to handle image preprocessing
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

# Train the model
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, save_loss_path=None, save_model_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f'Epoch {epoch+1}, Loss: {avg_train_loss}')

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f'Validation Loss: {avg_val_loss}, Accuracy: {100 * correct / total}%')

        # Save the model after each epoch
        if save_model_dir:
            model_path = os.path.join(save_model_dir, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), model_path)

    if save_loss_path:
        with open(save_loss_path, 'w') as f:
            for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
                f.write(f'Epoch {epoch+1}, Train Loss: {train_loss}, Validation Loss: {val_loss}\n')

    return avg_val_loss

# Main function
def main():
    parser = argparse.ArgumentParser(description='Train an MLP on the zh-plus/tiny-imagenet dataset.')
    parser.add_argument('--layer_count', type=int, default=2, help='Number of hidden layers (default: 2)')
    parser.add_argument('--width', type=int, default=512, help='Number of neurons per hidden layer (default: 512)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training (default: 8)')
    parser.add_argument('--save_model_dir', type=str, default='saved_models', help='Directory to save model checkpoints (default: saved_models)')
    args = parser.parse_args()

    # Load the zh-plus/tiny-imagenet dataset
    dataset = load_dataset('zh-plus/tiny-imagenet')

    # Split the dataset into train and validation sets
    train_dataset = dataset['train']
    val_dataset = dataset['valid']  # Assuming 'validation' is the correct key

    # Determine the number of classes
    num_classes = len(set(train_dataset['label']))

    # Determine the fixed resolution of the images
    image_size = 64  # Assuming the images are square

    # Define the model
    input_size = image_size * image_size  # Since images are grayscale
    hidden_sizes = [args.width] * args.layer_count
    output_size = num_classes

    model = MLP(input_size, hidden_sizes, output_size)

    # Create the directory to save models
    os.makedirs(args.save_model_dir, exist_ok=True)

    # Create DataLoader for training and validation
    train_loader = DataLoader(TinyImageNetDataset(train_dataset), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TinyImageNetDataset(val_dataset), batch_size=args.batch_size, shuffle=False)

    # Train the model and get the final loss
    save_loss_path = 'losses.txt'
    final_loss = train_model(model, train_loader, val_loader, save_loss_path=save_loss_path, save_model_dir=args.save_model_dir)

    # Calculate the number of parameters
    param_count = sum(p.numel() for p in model.parameters())

    # Create the folder for the model
    model_folder = f'mlp_model_l{args.layer_count}w{args.width}'
    os.makedirs(model_folder, exist_ok=True)

    # Save the final model
    model_path = os.path.join(model_folder, 'model.pth')
    torch.save(model.state_dict(), model_path)

    # Write the results to a text file in the model folder
    result_path = os.path.join(model_folder, 'results.txt')
    with open(result_path, 'w') as f:
        f.write(f'Layer Count: {args.layer_count}, Width: {args.width}, Parameter Count: {param_count}, Final Loss: {final_loss}\n')

    # Save a duplicate of the results in the 'results' folder
    results_folder = 'results'
    os.makedirs(results_folder, exist_ok=True)
    duplicate_result_path = os.path.join(results_folder, f'results_l{args.layer_count}w{args.width}.txt')
    with open(duplicate_result_path, 'w') as f:
        f.write(f'Layer Count: {args.layer_count}, Width: {args.width}, Parameter Count: {param_count}, Final Loss: {final_loss}\n')

if __name__ == '__main__':
    main()