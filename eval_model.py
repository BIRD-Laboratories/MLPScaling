import argparse
import torch
from datasets import load_dataset
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
from mmengine.model import BaseModel
from collections import OrderedDict
import re

# Custom Dataset class to handle image preprocessing
class TinyImageNetDataset(Dataset):
    def __init__(self, dataset, device='cpu'):
        self.dataset = dataset
        self.device = torch.device(device)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        img = example['image'].convert('RGB')
        img = np.array(img)
        img = img.astype(np.float32) / 255.0
        img = img.reshape(-1)
        img = torch.from_numpy(img).to(self.device)
        label = torch.tensor(example['label']).to(self.device)
        return img, label

# Define the MLP model
class MLP(BaseModel):
    def __init__(self, input_size, hidden_sizes, output_size, device='cpu'):
        super(MLP, self).__init__()
        self.device = torch.device(device)
        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            layers.append(torch.nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                layers.append(torch.nn.ReLU())
        self.model = torch.nn.Sequential(*layers).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, inputs, labels, mode='train'):
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        outputs = self.model(inputs)
        if mode == 'train':
            loss = self.criterion(outputs, labels)
            return dict(loss=loss)
        elif mode == 'val':
            loss = self.criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            total = labels.size(0)
            return dict(loss=loss, correct=correct, total=total)
        else:
            return outputs

    def val_step(self, data):
        inputs, labels = data
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        return {'loss': loss, 'correct': correct, 'total': total}

def get_model_config_from_state_dict(state_dict, output_size):
    # Extract all 'weight' keys
    weight_keys = [key for key in state_dict.keys() if key.endswith('.weight')]
    
    # Extract layer indices using regex
    layer_indices = []
    for key in weight_keys:
        match = re.search(r'model\.(\d+)\.weight', key)
        if match:
            layer_indices.append(int(match.group(1)))
    
    # Sort the layer indices
    layer_indices_sorted = sorted(layer_indices)
    
    # Get weights in order
    weights = [state_dict[f'model.{i}.weight'].shape for i in layer_indices_sorted]
    
    # Determine input_size, hidden_sizes, output_size
    input_size = weights[0][1]
    hidden_sizes = [w[0] for w in weights[:-1]]
    
    # Verify that the last layer's out_features match the output_size
    if weights[-1][0] != output_size:
        raise ValueError("Output size does not match the last layer's out_features.")
    
    return input_size, hidden_sizes, output_size

def main():
    parser = argparse.ArgumentParser(description='Evaluate an MLP on the zh-plus/tiny-imagenet dataset.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation (default: 8)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (default: cuda if available)')
    args = parser.parse_args()
    
    # Print parsed arguments
    print(f"Parsed arguments: {args}")
    
    # Load the zh-plus/tiny-imagenet dataset
    print("Loading the dataset...")
    dataset = load_dataset('zh-plus/tiny-imagenet')
    val_dataset = dataset['valid']
    print(f"Validation dataset loaded with {len(val_dataset)} samples.")
    
    # Determine the number of classes
    num_classes = len(set(val_dataset['label']))
    print(f"Number of classes: {num_classes}")
    
    # Load the checkpoint
    print("Loading the checkpoint...")
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    state_dict = checkpoint['state_dict']
    
    # Infer model configuration from state_dict
    print("Inferring model configuration...")
    input_size, hidden_sizes, output_size = get_model_config_from_state_dict(state_dict, num_classes)
    print(f"Model configuration: input_size={input_size}, hidden_sizes={hidden_sizes}, output_size={output_size}")
    
    # Create the model with inferred parameters
    print("Creating the model...")
    model = MLP(input_size, hidden_sizes, output_size, device=args.device)
    
    # Load the state_dict into the model
    print("Loading state_dict into the model...")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    
    # Set the model to evaluation mode
    print("Setting the model to evaluation mode.")
    model.eval()
    
    # Set up the data loader
    print("Setting up the data loader...")
    val_loader = DataLoader(TinyImageNetDataset(val_dataset, device=args.device), batch_size=args.batch_size, shuffle=False)
    
    # Evaluate the model
    print("Starting evaluation...")
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, labels = data
            print(f"Batch {i+1}: inputs shape={inputs.shape}, labels shape={labels.shape}")
            outputs = model.val_step(data)
            correct += outputs['correct']
            total += outputs['total']
            print(f"Batch {i+1}: Correct={outputs['correct']}, Total={outputs['total']}, Accuracy so far: {100 * correct / total:.2f}%")
    
    accuracy = 100 * correct / total
    print(f'Accuracy on the validation set: {accuracy:.2f}%')

if __name__ == '__main__':
    main()
