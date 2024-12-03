import os
import time
import json
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageNetKaggle(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}

        print(f"[{time.time()}] Starting to load class index mapping...")
        load_class_index_start = time.time()
        # Load class index mapping
        with open(os.path.join(root, "imagenet_class_index.json"), "r") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)
        load_class_index_end = time.time()
        print(f"[{time.time()}] Finished loading class index mapping. Duration: {load_class_index_end - load_class_index_start:.2f}s.")

        # Load validation labels if necessary
        if split == "val":
            print(f"[{time.time()}] Starting to load validation labels...")
            load_val_labels_start = time.time()
            with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "r") as f:
                self.val_to_syn = json.load(f)
            load_val_labels_end = time.time()
            print(f"[{time.time()}] Finished loading validation labels. Duration: {load_val_labels_end - load_val_labels_start:.2f}s.")

        samples_dir = os.path.join(root, "ILSVRC", "Data", "CLS-LOC", split)
        print(f"[{time.time()}] Starting to scan samples in {samples_dir}...")
        scan_start_time = time.time()

        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class.get(syn_id)
                if target is None:
                    continue
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn.get(entry)
                if syn_id is None:
                    continue
                target = self.syn_to_class.get(syn_id)
                if target is None:
                    continue
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)

        scan_end_time = time.time()
        print(f"[{time.time()}] Finished scanning samples. Total samples: {len(self.samples)}. Duration: {scan_end_time - scan_start_time:.2f}s.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]

# Training Configuration
batch_size = 128  # Set to 32, 64, or 128 as desired
num_workers = 8  # Set to 0 to disable prefetching and control data loading timing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[{time.time()}] {device} is used for training")

# Model, Loss, and Optimizer
print(f"[{time.time()}] Initializing model, criterion, and optimizer...")
model_init_start = time.time()
model = torchvision.models.resnet50(weights=None).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
model_init_end = time.time()
print(f"[{time.time()}] Model, criterion, and optimizer initialized. Duration: {model_init_end - model_init_start:.2f}s.")

# Data Transforms
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# Dataset and DataLoader
print(f"[{time.time()}] Creating dataset...")
dataset_creation_start = time.time()
dataset = ImageNetKaggle("/mnt/cephfs/imagenet/", "train", train_transform)
dataset_creation_end = time.time()
print(f"[{time.time()}] Dataset created. Duration: {dataset_creation_end - dataset_creation_start:.2f}s.")

print(f"[{time.time()}] Creating DataLoader...")
dataloader_creation_start = time.time()
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    pin_memory=True
)
dataloader_creation_end = time.time()
print(f"[{time.time()}] DataLoader created. Duration: {dataloader_creation_end - dataloader_creation_start:.2f}s.")

target_iteration = 5
data_loading_start_time = time.time()
for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Measure data transfer time (CPU to GPU)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if batch_idx +1 >= target_iteration:
            break

data_loading_end_time = time.time()
# data_transfer_time = data_transfer_end - data_transfer_start


inputs_list = []
targets_list = []
# output_list = []
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    pin_memory=True
)

model.to(device)            
model.train()
for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Measure data transfer time (CPU to GPU)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        inputs_list.append(inputs)
        targets_list.append(targets)
        if batch_idx <target_iteration:
            continue
        else:
            break



forward_backward_pass_start_time= time.time()
for i in range(len(inputs_list)):
    inputs = inputs_list [i]
    targets = targets_list[i]
    torch.cuda.synchronize()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    torch.cuda.synchronize()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

forward_backward_pass_end_time= time.time()

# backward_pass_start_time = time.time()
# for i in range(len(output_list)):
#     outputs = output_list[i]
#     targets = targets_list[i]
#     torch.cuda.synchronize()
#     loss = criterion(outputs, targets)
#     optimizer.zero_grad()
#     # Backward pass
#     torch.cuda.synchronize()
#     loss.backward()
#     optimizer.step()
#     torch.cuda.synchronize()
# backward_pass_end_time = time.time()

print(f"Data Loading Phase: Start Time = {data_loading_start_time}, "
      f"End Time = {data_loading_end_time}, "
      f"Duration = {round((data_loading_end_time - data_loading_start_time),2)}s")

print (f"ForwardBackward Pass phase: Start Time = {forward_backward_pass_start_time}, "
      f"End Time = {forward_backward_pass_end_time}, "
      f"Duration = {round((forward_backward_pass_end_time - forward_backward_pass_start_time),2)}s")

# print (f"Backward Pass phase: Start Time = {backward_pass_start_time}, "
#       f"End Time = {backward_pass_end_time}, "
#       f"Duration = {round((backward_pass_end_time - backward_pass_start_time),2)}s")