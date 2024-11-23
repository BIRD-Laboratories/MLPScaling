from modelscope.hub.api import HubApi
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
from mmengine.model import BaseModel
from mmengine.runner import Runner, EpochBasedTrainLoop, ValLoop
from mmengine.hooks import CheckpointHook, LoggerHook
from mmengine.optim import OptimWrapper

# Custom Dataset class to handle image preprocessing
class TinyImageNetDataset(Dataset):
    def __init__(self, dataset, device='cpu'):
        self.dataset = dataset
        self.device = torch.device(device)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        img = example['image']
        img = np.array(img.convert('L'))  # Convert PIL image to grayscale NumPy array
        img = img.reshape(-1)  # Flatten the image
        img = torch.from_numpy(img).float().to(self.device)  # Convert to tensor and move to device
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
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers).to(self.device)  # Move layers to device
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
    parser = argparse.ArgumentParser(description='Train an MLP on the zh-plus/tiny-imagenet dataset.')
    parser.add_argument('--layer_count', type=int, default=2, help='Number of hidden layers (default: 2)')
    parser.add_argument('--width', type=int, default=512, help='Number of neurons per hidden layer (default: 512)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training (default: 8)')
    parser.add_argument('--save_model_dir', type=str, default='saved_models', help='Directory to save model checkpoints (default: saved_models)')
    parser.add_argument('--access_token', type=str, help='ModelScope SDK access token (optional)')
    parser.add_argument('--upload_checkpoint', action='store_true', help='Upload checkpoint to ModelScope')
    parser.add_argument('--delete_checkpoint', action='store_true', help='Delete local checkpoint after uploading')
    args = parser.parse_args()

    # Load the zh-plus/tiny-imagenet dataset
    dataset = load_dataset('zh-plus/tiny-imagenet')

    # Split the dataset into train and validation sets
    train_dataset = dataset['train']
    val_dataset = dataset['valid']  # Correct key for validation set

    # Determine the number of classes
    num_classes = len(set(train_dataset['label']))

    # Determine the fixed resolution of the images
    image_size = 64  # Assuming the images are square

    # Define the model
    input_size = image_size * image_size  # Since images are grayscale
    hidden_sizes = [args.width] * args.layer_count
    output_size = num_classes

    train_cfg = dict(
      by_epoch=True,
      max_epochs=10,  # Set the number of epochs
      val_interval=1  # Perform validation every epoch
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(input_size, hidden_sizes, output_size, device=device)

    # Create the directory to save models
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
        val_dataloader=val_loader,
        val_cfg=dict(),
        default_hooks=dict(
            checkpoint=dict(
                type=CheckpointHook,
                interval=1,
                save_best=None,
                #max_keep=None,
                save_optimizer=False
            ),
            logger=dict(type=LoggerHook, interval=10),
        ),
    )

    # Start training
    runner.train()

    # Calculate the number of parameters
    param_count = sum(p.numel() for p in model.parameters())

    # Save results.txt in work_dir
    result_path = os.path.join(runner.work_dir, 'results.txt')
    with open(result_path, 'w') as f:
        f.write(f'Layer Count: {args.layer_count}, Width: {args.width}, Parameter Count: {param_count}\n')

    # Save a duplicate of the results in the 'results' folder
    results_folder = 'results'
    os.makedirs(results_folder, exist_ok=True)
    duplicate_result_path = os.path.join(results_folder, f'results_l{args.layer_count}_w{args.width}.txt')
    with open(duplicate_result_path, 'w') as f:
        f.write(f'Layer Count: {args.layer_count}, Width: {args.width}, Parameter Count: {param_count}\n')

    # Upload the entire folder to ModelScope if specified
    if args.upload_checkpoint:
        if not args.access_token:
            raise ValueError("Access token is required for uploading to ModelScope.")
        api = HubApi()
        api.login(args.access_token)
        api.push_model(
            model_id="puffy310/MLPScaling",
            model_dir=runner.work_dir  # Local model directory
        )

    # Delete the local model directory if specified
    if args.delete_checkpoint:
        import shutil
        shutil.rmtree(runner.work_dir)

if __name__ == '__main__':
    main()
