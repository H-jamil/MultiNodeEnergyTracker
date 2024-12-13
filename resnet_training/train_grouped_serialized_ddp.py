import os
import time
import json
import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import ImageFolder

def setup_distributed():
    """
    Initialize distributed training environment.
    """
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda", rank % torch.cuda.device_count())
    return device, rank, world_size

def cleanup_distributed():
    """
    Clean up distributed training environment.
    """
    dist.destroy_process_group()

class ResNet50System(nn.Module):
    def __init__(self):
        super(ResNet50System, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=False)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)

def get_dataset(data_path):
    """
    Create ImageNet dataset with necessary transformations.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = ImageFolder(root=data_path, transform=transform)
    return dataset

def group_batches(dataloader, group_size):
    """
    Generator that yields groups of batches from the dataloader.

    Args:
        dataloader (DataLoader): The DataLoader to group batches from.
        group_size (int): Number of batches per group.

    Yields:
        list: A list containing 'group_size' number of batches.
    """
    batch_group = []
    for batch in dataloader:
        batch_group.append(batch)
        if len(batch_group) == group_size:
            yield batch_group
            batch_group = []
    if batch_group:
        yield batch_group

def main():
    """
    Main training loop with serialized execution and timestamp logging.
    """
    # Setup distributed environment
    device, rank, world_size = setup_distributed()
    
    # Log rank information
    print(f"Rank {rank} initialized. World Size: {world_size}. Device: {device}")
    
    # Set seed for reproducibility (optional)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # Create dataset and dataloader
    data_path = '/home/cc/imagenet/ILSVRC/Data/CLS-LOC/train'  # Update if necessary
    dataset = get_dataset(data_path)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,  # Shuffling is handled by DistributedSampler
        num_workers=1,   # Increased from 1 to 4 for better data loading performance
        pin_memory=True,
        sampler=sampler
    )
    
    # Initialize model, move to device, and wrap with DDP
    model = ResNet50System().to(device)
    model = DDP(model, device_ids=[rank % torch.cuda.device_count()], output_device=rank % torch.cuda.device_count())
    
    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Define number of epochs
    num_epochs = 1
    
    # Define batch group size
    group_size = 5  # Number of batches per group
    
    # Initialize DataLoader iterator
    data_iter = iter(dataloader)
    
    # Training Loop
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Shuffle data differently every epoch
        print(f"Rank {rank}: Starting Epoch {epoch+1}/{num_epochs}")
        
        for group_idx, batch_group in enumerate(group_batches(dataloader, group_size), start=1):
            # ----------------------------
            # Data Loading (Disk to CPU + CPU to GPU)
            # ----------------------------
            data_load_start = time.time()
            
            # Initialize lists to hold GPU tensors
            images_list = []
            labels_list = []
            
            for batch in batch_group:
                images, labels = batch
                # Move data to GPU asynchronously
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                images_list.append(images)
                labels_list.append(labels)
            
            # Ensure all data transfers are complete
            torch.cuda.synchronize()
            data_load_end = time.time()
            
            # ----------------------------
            # Compute Forward, Backward Pass, and Optimization
            # ----------------------------
            compute_start = time.time()
            
            # Initialize cumulative loss for the group
            cumulative_loss = 0.0
            
            for images, labels in zip(images_list, labels_list):
                # Forward pass
                outputs = model(images)
                loss = model.module.criterion(outputs, labels)
                cumulative_loss += loss.item()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
            
            # Optimization step
            optimizer.step()
            
            # Ensure all CUDA operations are complete
            torch.cuda.synchronize()
            compute_end = time.time()
            
            # ----------------------------
            # Synchronization (All-Reduce)
            # ----------------------------
            sync_start = time.time()
            
            # Example synchronization: average cumulative loss across all processes
            avg_cumulative_loss = torch.tensor(cumulative_loss / group_size).to(device)
            dist.all_reduce(avg_cumulative_loss, op=dist.ReduceOp.SUM)
            avg_cumulative_loss /= world_size
            
            sync_end = time.time()
            
            # ----------------------------
            # Logging
            # ----------------------------
            batch_range_start = (group_idx - 1) * group_size + 1
            batch_range_end = group_idx * group_size
            print(f"Rank {rank}, Epoch {epoch}, Batches {batch_range_start}-{batch_range_end}:")
            print(f"    Data Load: start {data_load_start:.4f} end {data_load_end:.4f} total {data_load_end - data_load_start:.4f} seconds")
            print(f"    Compute: start {compute_start:.4f} end {compute_end:.4f}  total {compute_end - compute_start:.4f} seconds")
            print(f"    Synchronization: start {sync_start:.4f} end {sync_end:.4f} total {sync_end - sync_start:.4f} seconds")
            print(f"    Average Loss: {avg_cumulative_loss.item():.4f}")
    
    # Clean up distributed environment
    cleanup_distributed()
    print(f"Rank {rank}: Training completed.")

if __name__ == "__main__":
    main()
