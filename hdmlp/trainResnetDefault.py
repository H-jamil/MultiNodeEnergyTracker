import os
import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def main():
    # Set your ImageNet training directory.
    # data_dir = "/home/cc/train/ILSVRC/Data/CLS-LOC/train"
    data_dir = "/home/cc/train/subset/train"
    
    # Job configuration parameters.
    batch_size = 32       # Global batch size
    drop_last = False     # Do not drop the last incomplete batch
    
    # Define transforms:
    # 1. Resize the image to 224x224.
    # 2. Convert the image to a tensor.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Create the dataset using torchvision's ImageFolder.
    # It expects the data to be organized in subdirectories per class.
    print("Creating dataset...")
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    print("Dataset created")
    
    # Create the DataLoader.
    print("Creating DataLoader...")
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=drop_last)
    print("DataLoader created")
    
    # Use CUDA if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Iterate over a few batches to test I/O.
    start_time = time.time()
    for batch_idx, (images, labels) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if batch_idx % 5 == 0:
            print(f"Batch {batch_idx}: images shape = {images.shape}, labels shape = {labels.shape}")
        # For testing, break after a few batches (optional)
        # if batch_idx >= 5:
        #     break
    elapsed = time.time() - start_time
    print(f"Took {elapsed:.2f} seconds")
    
if __name__ == "__main__":
    main()
