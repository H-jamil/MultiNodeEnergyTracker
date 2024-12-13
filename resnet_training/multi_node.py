import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl

class ResNet50System(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained=False)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

def get_dataset():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = torchvision.datasets.ImageFolder(
        root='/home/cc/imagenet/ILSVRC/Data/CLS-LOC/train',
        transform=transform
    )
    return dataset

def main(rank, world_size):
    # Initialize distributed training
    os.environ['MASTER_ADDR'] = '10.52.0.10'  # Replace with actual IP
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Create dataset and dataloader
    dataset = get_dataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=8,
        sampler=sampler
    )
    
    # Initialize model and trainer
    model = ResNet50System()
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],  # assuming one GPU per node
        num_nodes=2,
        strategy='ddp',
        max_epochs=2
    )
    
    # Train
    trainer.fit(model, dataloader)
    
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2  # total number of processes
    rank = int(os.environ['RANK'])  # process rank
    main(rank, world_size)