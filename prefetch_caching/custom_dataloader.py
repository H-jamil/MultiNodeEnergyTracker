import os
import json
import torch
import time
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from PIL import Image

class ImageNetDataset(Dataset):
    def __init__(self, root_dir, json_path, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        with open(json_path) as f:
            self.class_idx = json.load(f)
        
        self.class_to_idx = {cls: int(idx) for idx, (cls, _) in self.class_idx.items()}
        
        classes = sorted(os.listdir(root_dir))
        for cls in classes:
            cls_dir = os.path.join(root_dir, cls)
            for img_name in sorted(os.listdir(cls_dir)):
                self.samples.append((
                    os.path.join(cls_dir, img_name),
                    self.class_to_idx[cls]
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return idx, img, label
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            return idx, torch.randn(3, 224, 224), -1

class DeterministicSampler(Sampler):
    def __init__(self, dataset, num_replicas, rank, seed=42):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        return iter(indices[self.rank::self.num_replicas])

    def __len__(self):
        return (len(self.dataset) + self.num_replicas - 1) // self.num_replicas

def setup(rank, world_size):
    device_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )
    return device_id

def train(rank, world_size, data_dir, json_path, batch_size=32, num_epochs=2, seed=42):
    device_id = setup(rank, world_size)
    
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset and sampler
    dataset = ImageNetDataset(
        root_dir=data_dir,
        json_path=json_path,
        transform=transform
    )
    sampler = DeterministicSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank,
        seed=seed
    )
    
    # Dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True  # Important for equal batch sizes across GPUs
    )
    
    # Model with DDP
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 112 * 112, 1000)
    ).to(device_id)
    model = DDP(model, device_ids=[device_id])
    
    optimizer = optim.SGD(model.parameters(), lr=0.001 * world_size)  # Scaled learning rate
    loss_fn = nn.CrossEntropyLoss()
    
    # Plan file
    # plan_file = open(f'plan_rank{rank}_1.txt', 'w')
    
    # Training loop
    start_time = time.time()
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        model.train()
        
        for batch_idx, (indices, data, targets) in enumerate(dataloader):
            data = data.to(device_id, non_blocking=True)
            targets = targets.to(device_id, non_blocking=True)
            
            # Filter invalid samples
            valid_mask = targets != -1
            if valid_mask.sum() == 0:
                continue
                
            data = data[valid_mask]
            targets = targets[valid_mask]
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Synchronize all processes
            torch.cuda.synchronize()
            
            # Logging
            # if rank == 0:  # Only master logs
            print(f"Epoch {epoch}/{num_epochs} Step {batch_idx}/{len(dataloader)} Loss: {loss.item():.4f}")
            if batch_idx >= 10:
                    print(f"Time taken for training is {time.time()-start_time} seconds")
                    break
            # # Write to plan file
    #         files = [dataset.samples[i][0] for i in indices.tolist()]
    #         plan_entry = (f"Rank {rank}, Epoch {epoch}, Step {batch_idx}:\n" +
    #                       "\n".join(files) + "\n\n")
    #         plan_file.write(plan_entry)
    
    # plan_file.close()
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, required=True)
    args = parser.parse_args()
    
    train(
        rank=args.rank,
        world_size=2,
        data_dir='/home/cc/train/ILSVRC/Data/CLS-LOC/train',
        json_path='/home/cc/train/imagenet_class_index.json',
        batch_size=32,
        num_epochs=1,
        seed=42
    )