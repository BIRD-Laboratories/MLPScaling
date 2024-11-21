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

# Define the MLP model
class MLP(BaseModel):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, labels, mode='train'):
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

# Define the training loop
class MLPTrainLoop(EpochBasedTrainLoop):
    def run_iter(self, data_batch: dict, train_mode: bool = True) -> None:
        data_batch = self.data_preprocessor(data_batch, training=True)
        outputs = self.model(**data_batch, mode='train')
        parsed_outputs = self.model.parse_losses(outputs)
        self.optim_wrapper.update_params(parsed_outputs)

# Define the validation loop
class MLPValLoop(ValLoop):
    def run_iter(self, data_batch: dict, train_mode: bool = False) -> None:
        data_batch = self.data_preprocessor(data_batch, training=False)
        outputs = self.model(**data_batch, mode='val')
        self.outputs['loss'].append(outputs['loss'].item())
        self.outputs['correct'].append(outputs['correct'])
        self.outputs['total'].append(outputs['total'])

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

    # Set up Git to use hf-mirror as a proxy
    os.environ['GIT_PROXY_COMMAND'] = 'proxychains4 git'

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

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Define the runner
    runner = Runner(
        model=model,
        work_dir=args.save_model_dir,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optim_wrapper=dict(optimizer=optimizer),
        train_loop=MLPTrainLoop,
        val_loop=MLPValLoop,
        val_interval=1,
        default_hooks=dict(
            checkpoint=dict(type=CheckpointHook, interval=1, save_best='auto') if not args.delete_checkpoint else None,
            logger=dict(type=LoggerHook, interval=10)
        )
    )

    # Start training
    runner.train()

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
        f.write(f'Layer Count: {args.layer_count}, Width: {args.width}, Parameter Count: {param_count}\n')

    # Save a duplicate of the results in the 'results' folder
    results_folder = 'results'
    os.makedirs(results_folder, exist_ok=True)
    duplicate_result_path = os.path.join(results_folder, f'results_l{args.layer_count}w{args.width}.txt')
    with open(duplicate_result_path, 'w') as f:
        f.write(f'Layer Count: {args.layer_count}, Width: {args.width}, Parameter Count: {param_count}\n')

    # Upload the model to ModelScope if specified
    if args.upload_checkpoint:
        if not args.access_token:
            raise ValueError("Access token is required for uploading to ModelScope.")
        api = HubApi()
        api.login(args.access_token)
        api.push_model(
            model_id="puffy310/MLPScaling", 
            model_dir=model_folder  # Local model directory, the directory must contain configuration.json
        )

    # Delete the local model directory if specified
    if args.delete_checkpoint:
        import shutil
        shutil.rmtree(model_folder)

if __name__ == '__main__':
    main()
