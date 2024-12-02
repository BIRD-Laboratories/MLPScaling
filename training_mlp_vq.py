import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from mmengine.model import BaseModel
from mmengine.runner import Runner
from mmengine.optim import OptimWrapper
from sklearn.model_selection import train_test_split

# Experimental
# Custom Dataset class to load codebooks and labels from the dataset
class TinyImageNetDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        # Ensure each example has 'codebook' and 'label' keys
        codebook = example['codebook']
        label = example['label']
        
        # Convert codebook to float tensor
        codebook = torch.tensor(codebook, dtype=torch.float32)
        # Convert label to int tensor
        label = torch.tensor(label, dtype=torch.int64)
        
        return codebook, label

# Define the MLP model
class MLP(BaseModel):
    def __init__(self, input_size, hidden_sizes, output_size, device='cpu'):
        super(MLP, self).__init__()
        self.device = torch.device(device)
        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
    
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
    
    def train_step(self, data, optim_wrapper):
        inputs, labels = data
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        optim_wrapper.update_params(loss)
        return {'loss': loss}
    
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

# Main function
def main():
    parser = argparse.ArgumentParser(description='Train an MLP on preprocessed codebook vectors.')
    parser.add_argument('--layer_count', type=int, default=2, help='Number of hidden layers (default: 2)')
    parser.add_argument('--width', type=int, default=512, help='Number of neurons per hidden layer (default: 512)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training (default: 8)')
    parser.add_argument('--save_model_dir', type=str, default='models', help='Directory to save model checkpoints (default: models)')
    args = parser.parse_args()
    
    # Load the encoded dataset
    with open('dataset/encoded_tiny_imagenet.json', 'r') as f:
        encoded_data = json.load(f)
    
    # Split the data into training and validation sets
    train_data, val_data = train_test_split(encoded_data, test_size=0.2, random_state=42)
    
    # Determine the number of classes
    labels = [example['label'] for example in train_data]
    num_classes = len(set(labels))
    
    # Define the model
    input_size = 256  # Dimension of the encoder's pooler_output for ViT-base
    hidden_sizes = [args.width] * args.layer_count
    output_size = num_classes
    
    train_cfg = dict(
        by_epoch=True,
        max_epochs=10,  # Set the number of epochs
        val_interval=1  # Perform validation every epoch
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(input_size, hidden_sizes, output_size, device=device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    optim_wrapper = OptimWrapper(optimizer=optimizer)
    
    # Set up data loaders
    train_loader = DataLoader(
        TinyImageNetDataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4  # Adjust based on system capabilities
    )
    
    val_loader = DataLoader(
        TinyImageNetDataset(val_data),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )
    
    # Define the folder name based on model layers and width
    model_folder = f"mlp_l{args.layer_count}_w{args.width}"
    model_folder_path = os.path.join(args.save_model_dir, model_folder)
    os.makedirs(model_folder_path, exist_ok=True)
    
    # Define the runner with the new work_dir and modified CheckpointHook
    runner = Runner(
        model=model,
        work_dir=model_folder_path,
        train_dataloader=train_loader,
        optim_wrapper=optim_wrapper,
        train_cfg=train_cfg,
        default_hooks=dict(
            checkpoint=dict(
                type=CheckpointHook,
                interval=1,
                save_best=None,
                save_optimizer=False
            ),
            logger=dict(type=LoggerHook, interval=10),
        ),
    )
    
    # Start training
    runner.train()

if __name__ == '__main__':
    main()


