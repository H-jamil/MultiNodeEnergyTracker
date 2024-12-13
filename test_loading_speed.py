import os
import json
import time
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class RamdiskDataset(Dataset):
    def __init__(self, json_file, transform=None):
        with open(json_file, 'r') as f:
            self.samples = json.load(f)
            # self.samples = self.samples[:20000]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

class CephFSDataset(Dataset):
    def __init__(self, json_file, transform=None):
        with open(json_file, 'r') as f:
            data = json.load(f)
        self.samples = data['selected_samples']
        # self.samples = self.samples[:20000]
        # self.labels = data['selected_labels']  # Not used in loading speed test
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# Data Transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

def profile_data_loading(dataloader, device, description=""):
    # start_time = time.perf_counter()
    Epoch = 8
    start_time = time.time()
    for _ in range(0,Epoch):
    # print(start_time)
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(device, non_blocking=True)
            else:
                inputs = batch.to(device, non_blocking=True)
    # end_time = time.perf_counter()
    end_time = time.time()
    total_time = end_time - start_time
    print(f"{description} Data Loading start {start_time:.4f} end {end_time:.4f} total Time: {total_time:.2f} seconds")
    return total_time

def main():
    # Paths to JSON files
    ramdisk_json_path = "/home/cc/MultiNodeEnergyTracker/copied_files_20241209_220407_file_paths.json"  # Update with your actual path
    cephfs_json_path = "/mnt/mycephfs/imagenet/selected_files_20GB_train.json"    # Update with your actual path

    # Create Dataset instances
    ramdisk_dataset = RamdiskDataset(json_file=ramdisk_json_path, transform=transform)
    cephfs_dataset = CephFSDataset(json_file=cephfs_json_path, transform=transform)

    # DataLoader parameters
    batch_size = 128
    num_workers = 8  # Adjust based on your CPU cores
    pin_memory = True  # If using GPU

    # Create DataLoader instances
    ramdisk_dataloader = DataLoader(
        ramdisk_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory
    )

    cephfs_dataloader = DataLoader(
        cephfs_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory
    )

    # Determine the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type != 'cuda':
        print("CUDA is not available. Exiting profiling.")
        return

    # Warm-up runs
    # print("Running warm-up for Ramdisk...")
    # profile_data_loading(ramdisk_dataloader, device, description="Ramdisk (Warm-up)")
    # print("Running warm-up for CephFS...")
    # profile_data_loading(cephfs_dataloader, device, description="CephFS (Warm-up)")

    # Actual profiling runs
    print("\nStarting profiling runs...")

    # Profile Ramdisk DataLoader
    ramdisk_time = profile_data_loading(ramdisk_dataloader, device, description="Ramdisk")

    # Profile CephFS DataLoader
    cephfs_time = profile_data_loading(cephfs_dataloader, device, description="CephFS")

    # Summary
    print("\n--- Profiling Summary ---")
    print(f"Ramdisk Data Loading Time: {ramdisk_time:.2f} seconds")
    print(f"CephFS Data Loading Time: {cephfs_time:.2f} seconds")

    # Calculate speedup
    if cephfs_time > 0:
        speedup = cephfs_time / ramdisk_time
        print(f"Speedup (Ramdisk vs. CephFS): {speedup:.2f}x faster")
    else:
        print("CephFS loading time is too low to calculate speedup.")

if __name__ == "__main__":
    main()
