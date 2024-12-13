import os
import time
import json
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Set seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ImageNetKaggle(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}

        print(f"[{time.time()}] Starting to load class index mapping...")
        load_class_index_start = time.time()
        with open(os.path.join(root, "imagenet_class_index.json"), "r") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)
        load_class_index_end = time.time()
        print(f"[{time.time()}] Finished loading class index mapping. Duration: {load_class_index_end - load_class_index_start:.2f}s.")

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

        # Sort samples for deterministic ordering
        sorted_pairs = sorted(zip(self.samples, self.targets), key=lambda x: x[0])
        self.samples, self.targets = zip(*sorted_pairs)
        self.samples = list(self.samples)
        self.targets = list(self.targets)

        scan_end_time = time.time()
        print(f"[{time.time()}] Finished scanning samples. Total samples: {len(self.samples)}. Duration: {scan_end_time - scan_start_time:.2f}s.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]

# Set seed for reproducibility
set_seed(42)

# Training Configuration
batch_size = 128
num_workers = 8
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
dataset = ImageNetKaggle("/mnt/cached_data/imagenet", "train", train_transform)
# dataset = ImageNetKaggle("/mnt/mycephfs/imagenet/", "train", train_transform)
dataset_creation_end = time.time()
print(f"[{time.time()}] Dataset created. Duration: {dataset_creation_end - dataset_creation_start:.2f}s.")

print(f"[{time.time()}] Creating DataLoader...")
dataloader_creation_start = time.time()
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,  # Disabled shuffling for deterministic ordering
    pin_memory=True
)
dataloader_creation_end = time.time()
print(f"[{time.time()}] DataLoader created. Duration: {dataloader_creation_end - dataloader_creation_start:.2f}s.")

Epoch = 16
target_iteration = 5
data_loading_start_time = time.time()
for _ in range(0,Epoch):
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if batch_idx + 1 >= target_iteration:
            break

data_loading_end_time = time.time()

inputs_list = []
targets_list = []
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,  # Disabled shuffling for deterministic ordering
    pin_memory=True
)

model.to(device)            
model.train()
for batch_idx, (inputs, targets) in enumerate(dataloader):
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    inputs_list.append(inputs)
    targets_list.append(targets)
    if batch_idx < target_iteration:
        continue
    else:
        break

forward_backward_pass_start_time = time.time()
for i in range(len(inputs_list)):
    inputs = inputs_list[i]
    targets = targets_list[i]
    torch.cuda.synchronize()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    torch.cuda.synchronize()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

forward_backward_pass_end_time = time.time()

print(f"Data Loading Phase: Start Time = {data_loading_start_time}, "
      f"End Time = {data_loading_end_time}, "
      f"Duration = {round((data_loading_end_time - data_loading_start_time),2)}s")

print(f"ForwardBackward Pass phase: Start Time = {forward_backward_pass_start_time}, "
      f"End Time = {forward_backward_pass_end_time}, "
      f"Duration = {round((forward_backward_pass_end_time - forward_backward_pass_start_time),2)}s")