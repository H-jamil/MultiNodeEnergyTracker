import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import argparse
import datetime
from collections import defaultdict, OrderedDict

# 1. Enhanced Plan File Parser with Validation
def parse_plan_file(plan_path: str) -> dict:
    plan = defaultdict(lambda: defaultdict(list))
    current_epoch = None
    current_step = None
    
    try:
        with open(plan_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith("Rank"):
                    parts = line.split(',')
                    if len(parts) < 3:
                        continue
                    try:
                        current_epoch = int(parts[1].split()[-1])
                        current_step = int(parts[2].split()[1].strip(':'))
                    except (IndexError, ValueError) as e:
                        print(f"Invalid plan entry: {line}")
                        continue
                elif current_epoch is not None and current_step is not None:
                    plan[current_epoch][current_step].append(line)
    
    except Exception as e:
        print(f"Error parsing plan file: {str(e)}")
        raise
    
    # Validate plan structure
    for epoch in plan:
        steps = sorted(plan[epoch].keys())
        if steps != list(range(len(steps))):
            raise ValueError(f"Non-consecutive steps in epoch {epoch}")
    
    return plan

# 2. Robust Dataset Class with Synchronization
class PlanDataset(Dataset):
    def __init__(self, plan_path, transform=None):
        self.plan = parse_plan_file(plan_path)
        self.transform = transform
        
        # Create validated step list
        self.steps = []
        for epoch in sorted(self.plan.keys()):
            steps = sorted(self.plan[epoch].keys())
            if steps != list(range(len(steps))):
                raise ValueError(f"Non-consecutive steps in epoch {epoch}")
            self.steps.extend([(epoch, step) for step in steps])

        # Verify steps exist
        if not self.steps:
            raise ValueError("No valid steps found in plan file")

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, idx):
        try:
            epoch, step = self.steps[idx]
            paths = self.plan[epoch][step]
            
            if not paths:
                return torch.zeros(0), torch.zeros(0)
            
            batch = []
            for path in paths:
                try:
                    img = Image.open(path).convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    batch.append(img)
                except Exception as e:
                    print(f"Error loading {path}: {str(e)}")
                    batch.append(torch.zeros(3, 224, 224))
            
            return torch.stack(batch), torch.zeros(len(batch))
        except IndexError:
            print(f"Invalid index {idx} in dataset")
            return torch.zeros(0), torch.zeros(0)

# 3. DDP Setup with Enhanced Error Handling
def setup(rank, world_size):
    try:
        # Clear existing process group
        if dist.is_initialized():
            dist.destroy_process_group()
            
        # Set device
        device_id = rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        
        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=60))
            
        return device_id
    except Exception as e:
        print(f"Distributed setup failed on rank {rank}: {str(e)}")
        raise

# 4. Training Loop with Synchronization
def main(rank, world_size):
    device_id = setup(rank, world_size)
    print(f"Rank {rank} initialized on device {device_id}")

    # Training parameters
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    try:
        # Dataset validation
        dataset = PlanDataset(f"plan_rank{rank}.txt", transform)
        if len(dataset) == 0:
            raise ValueError(f"Rank {rank} has empty dataset")
            
        # DataLoader with synchronized batch counts
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=True  # Ensure consistent batch counts
        )
        
        # Model setup
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 112 * 112, 1000)
        ).to(device_id)
        model = DDP(model, device_ids=[device_id])
        
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        
        # Training loop with barrier synchronization
        start_time = time.time()
        for epoch in range(1):
            model.train()
            dist.barrier()  # Sync before epoch
            
            for batch_idx, (data, _) in enumerate(loader):
                dist.barrier()  # Sync before batch
                
                if data.shape[0] == 0:
                    continue
                
                data = data.squeeze(0).to(device_id)
                optimizer.zero_grad()
                outputs = model(data)
                loss = outputs.mean()
                loss.backward()
                optimizer.step()
                
                # if rank == 0:
                print(f"Epoch {epoch} Step {batch_idx} Loss: {loss.item()}")
                
                dist.barrier()  # Sync after batch
                
                if batch_idx >= 10:
                    print(f"Time taken for training is {time.time()-start_time} seconds")
                    break
            
            dist.barrier()  # Sync after epoch
        
        if rank == 0:
            torch.save(model.state_dict(), "model.pth")
            
    except Exception as e:
        print(f"Rank {rank} failed: {str(e)}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world_size', type=int, default=2)
    args = parser.parse_args()
    
    main(args.rank, args.world_size)