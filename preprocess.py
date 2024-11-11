## Debugging, don't use.

import os
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image

# Preprocess the images
def preprocess_image(example, image_size):
    image = example['image'].convert('RGB')  # Directly use the PIL image
    transform = transforms.Compose([
        #transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image)
    return {'image': image, 'label': example['label']}

# Main function
def main():
    # Load the dataset
    dataset = load_dataset('zh-plus/tiny-imagenet')
    train_dataset = dataset['train']
    val_dataset = dataset['valid']

    # Determine the fixed resolution of the images
    example_image = train_dataset[0]['image']  # Directly use the PIL image
    image_size = example_image.size[0]  # Assuming the images are square

    # Preprocess the dataset
    train_dataset = train_dataset.map(lambda x: preprocess_image(x, image_size))
    val_dataset = val_dataset.map(lambda x: preprocess_image(x, image_size))

    # Save the preprocessed datasets
    train_dataset.save_to_disk('preprocessed_train_dataset')
    val_dataset.save_to_disk('preprocessed_val_dataset')

if __name__ == '__main__':
    main()