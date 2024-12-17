import os
import time
import json
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import sys
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import shutil

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_json_dataset(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        # Ensure both lists exist and have the same length
        samples = data.get('selected_samples', [])
        targets = data.get('selected_targets', [])
        if len(samples) != len(targets):
            print("Error: 'selected_samples' and 'selected_targets' lists must be of the same length.")
            sys.exit(1)
        return samples, targets
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        sys.exit(1)

class ImageNetKaggle(Dataset):
    def __init__(self, root, split, ceph_mount=None, ramfs_mount=None, json_=None, json_file=None, transform=None):
        """
        Args:
            root (str): Root directory of the dataset.
            split (str): One of 'train' or 'val'.
            json_file (str, optional): Path to the JSON file containing image paths and labels.
                                        If None, the dataset is loaded in directory-based mode.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        self.json_file = json_file
        self.json = json_ 
        self.ceph_mount= ceph_mount
        self.ramfs_mount = ramfs_mount

        print(f"[{time.time()}] Starting to load class index mapping...")
        load_class_index_start = time.time()
        # Load class index mapping
        class_index_path = os.path.join(root, "imagenet_class_index.json")
        if not os.path.exists(class_index_path):
            print(f"Class index file not found at {class_index_path}.")
            sys.exit(1)
        with open(class_index_path, "r") as f:
            json_file_class = json.load(f)
            for class_id, v in json_file_class.items():
                self.syn_to_class[v[0]] = int(class_id)
        load_class_index_end = time.time()
        print(f"[{time.time()}] Finished loading class index mapping. Duration: {load_class_index_end - load_class_index_start:.2f}s.")

        if split == "val":
            print(f"[{time.time()}] Starting to load validation labels...")
            load_val_labels_start = time.time()
            val_labels_path = os.path.join(root, "ILSVRC2012_val_labels.json")
            if not os.path.exists(val_labels_path):
                print(f"Validation labels file not found at {val_labels_path}.")
                sys.exit(1)
            with open(val_labels_path, "r") as f:
                self.val_to_syn = json.load(f)
            load_val_labels_end = time.time()
            print(f"[{time.time()}] Finished loading validation labels. Duration: {load_val_labels_end - load_val_labels_start:.2f}s.")

        if self.json_file:
            print(f"[{time.time()}] Loading dataset from JSON file: {self.json_file}")
            self.load_from_json()
        else:
            # print(f"[{time.time()}] Loading dataset from directory structure.")
            # self.load_from_directory(root, split)
            samples_list, targets_list = load_json_dataset(self.json)
            ramfs_files = [fp.replace(self.ceph_mount, self.ramfs_mount) for fp in samples_list]
            self.samples = ramfs_files
            self.targets = targets_list

    def load_from_json(self):
        """
        Loads samples and targets from a JSON file.
        Assumes 'selected_samples' and 'selected_targets' are two separate lists of the same length.
        """
        samples_list, targets_list = load_json_dataset(self.json_file)
        self.samples = samples_list
        self.targets = targets_list
        # for img_path, label in tqdm(zip(samples_list, targets_list), total=len(samples_list), desc="Loading samples from JSON"):
        #     # Ensure the image path exists
        #     # if not os.path.isfile(img_path):
        #     #     print(f"Image file not found: {img_path}. Skipping.")
        #     #     continue
        #     # # Validate the label
        #     # if label not in self.syn_to_class.values():
        #     #     print(f"Label '{label}' not found in class index mapping. Skipping image: {img_path}.")
        #     #     continue
        #     self.samples.append(img_path)
        #     self.targets.append(label)

    def load_from_directory(self, root, split):
        """
        Loads samples and targets by scanning directories.
        """
        samples_dir = os.path.join(root, "ILSVRC", "Data", "CLS-LOC", split)
        if not os.path.exists(samples_dir):
            print(f"Samples directory not found at {samples_dir}.")
            sys.exit(1)
        print(f"[{time.time()}] Starting to scan samples in {samples_dir}...")
        scan_start_time = time.time()

        if split == "train":
            for syn_id in tqdm(os.listdir(samples_dir), desc="Scanning train classes"):
                syn_folder = os.path.join(samples_dir, syn_id)
                if not os.path.isdir(syn_folder):
                    continue
                target = self.syn_to_class.get(syn_id)
                if target is None:
                    print(f"Synset ID '{syn_id}' not found in class index mapping.")
                    continue
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    if not os.path.isfile(sample_path):
                        continue
                    self.samples.append(sample_path)
                    self.targets.append(target)
        elif split == "val":
            for entry in tqdm(os.listdir(samples_dir), desc="Scanning val images"):
                sample_path = os.path.join(samples_dir, entry)
                if not os.path.isfile(sample_path):
                    continue
                syn_id = self.val_to_syn.get(entry)
                if syn_id is None:
                    print(f"Synset ID for validation image '{entry}' not found.")
                    continue
                target = self.syn_to_class.get(syn_id)
                if target is None:
                    print(f"Synset ID '{syn_id}' not found in class index mapping.")
                    continue
                self.samples.append(sample_path)
                self.targets.append(target)

        scan_end_time = time.time()
        print(f"[{time.time()}] Finished scanning samples. Total samples: {len(self.samples)}. Duration: {scan_end_time - scan_start_time:.2f}s.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range.")
        image_path = self.samples[idx]
        label = self.targets[idx]
        try:
            x = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            # Optionally, handle corrupted images by returning a dummy image or skipping
            x = Image.new("RGB", (224, 224), (0, 0, 0))  # Example: black image
        if self.transform:
            x = self.transform(x)
        return x, label

def parse_arguments():
    parser = argparse.ArgumentParser(description="Compare CephFS and Ramfs (ramdisk) reading times with prefetching.")
    parser.add_argument(
        "filesystem",
        choices=["cephfs", "ramfs"],
        help="Choose which filesystem to test: 'cephfs' or 'ramfs'."
    )
    parser.add_argument(
        "--json_file",
        type=str,
        required=True,
        help="Path to the JSON file containing 'selected_samples'."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        choices=range(1, 17),
        metavar="[1-16]",
        help="Number of threads to use for loading/prefetching files (1 to 16). Default is 4."
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=5,
        help="Number of files to group together for reading/prefetching. Default is 5."
    )
    parser.add_argument(
        "--ceph_mount",
        type=str,
        default="/mnt/mycephfs",
        help="Mount point for CephFS. Default is /mnt/mycephfs."
    )
    parser.add_argument(
        "--ramdisk",
        type=str,
        default="/mnt/ramdisk",
        help="Mount point for ramdisk (ramfs). Default is /mnt/ramdisk."
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=1,
        help="epoch for data loading"
    )
    return parser.parse_args()

def load_json(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data['selected_samples']
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        sys.exit(1)

def read_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            f.read()
        return True
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

def prefetch_file(file_path, ceph_mount, ramdisk):
    try:
        rel_path = os.path.relpath(file_path, ceph_mount)
        dest_path = os.path.join(ramdisk, rel_path)
        dest_dir = os.path.dirname(dest_path)
        os.makedirs(dest_dir, exist_ok=True)
        if not os.path.exists(dest_path):
            shutil.copy2(file_path, dest_path)
        return True
    except Exception as e:
        print(f"Error prefetching {file_path}: {e}")
        return False

def read_files_concurrently(file_paths, num_threads):
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(read_file, fp): fp for fp in file_paths}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Reading files"):
            success = future.result()
            if not success:
                print(f"Failed to read {futures[future]}")
    end_time = time.time()
    return end_time - start_time

def prefetch_files_concurrently(file_paths, num_threads, ceph_mount, ramdisk):
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(prefetch_file, fp, ceph_mount, ramdisk): fp for fp in file_paths}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Prefetching files"):
            success = future.result()
            if not success:
                print(f"Failed to prefetch {futures[future]}")
    end_time = time.time()
    return start_time,end_time,end_time - start_time


def clear_system_cache():
    try:
        subprocess.run(['sudo', 'sync'], check=True)
        subprocess.run(['sudo', 'sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'], check=True)
        print("System cache cleared.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to clear cache: {e}")
        sys.exit(1)

def clear_directory():
    try:
        # subprocess.run(['sudo', 'sync'], check=True)
        subprocess.run(['sudo', 'rm', '-r', '/mnt/ramdisk/imagenet/ILSVRC'], check=True)
        print("Directory has been cleared")
    except subprocess.CalledProcessError as e:
        print(f"Failed to clear Directory: {e}")
        sys.exit(1)


def check_mount(mount_point):
    return os.path.ismount(mount_point)

def verify_prefetching(file_group, ceph_mount, ramdisk):
    """Verify that files are present in the ramdisk after prefetching and are regular files."""
    all_present = True
    for file_path in file_group:
        rel_path = os.path.relpath(file_path, ceph_mount)
        ramfs_path = os.path.join(ramdisk, rel_path)
        if not os.path.exists(ramfs_path):
            print(f"File not found in ramdisk: {ramfs_path}")
            all_present = False
        elif not os.path.isfile(ramfs_path):
            print(f"Non-regular file in ramdisk: {ramfs_path}")
            all_present = False
    return all_present


def main():
    args = parse_arguments()
    # Validate mount points
    print(f"{args.filesystem} is getting tested")
    if args.filesystem == "cephfs":
        if not check_mount(args.ceph_mount):
            print(f"CephFS mount point '{args.ceph_mount}' is not mounted.")
            sys.exit(1)
    elif args.filesystem == "ramfs":
        if not check_mount(args.ceph_mount):
            print(f"CephFS mount point '{args.ceph_mount}' is not mounted.")
            sys.exit(1)
        if not check_mount(args.ramdisk):
            print(f"Ramdisk mount point '{args.ramdisk}' is not mounted.")
            sys.exit(1)
    selected_files = load_json(args.json_file)
    total_files = len(selected_files)
    print(f"Total files to process: {total_files}")
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
    if args.filesystem == "cephfs":
        clear_system_cache()
        root_dir = "/mnt/mycephfs/imagenet"
        json_path = "/mnt/mycephfs/imagenet/selected_files_20GB_train.json"
        # Initialize the Dataset in 'ramfs' mode
        dataset_cephfs = ImageNetKaggle(
            root=root_dir,
            split="train",  # or "val" based on your JSON file
            json_file=json_path,
            transform=train_transform
        )
        dataloader_cephfs = DataLoader(
        dataset_cephfs,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # Adjust based on your system's capabilities
        pin_memory=True
        )
        print(f"[{time.time()}] Finished creating dataset...")
        print(f"Data loading started .... ")
        data_loading_start_time = time.time()
        for _ in range (0, args.epoch):
            for batch_idx, (images, labels) in enumerate(dataloader_cephfs):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
        data_loading_end_time = time.time()
        print(f"Data loading start at {data_loading_start_time} and end {data_loading_end_time} for {data_loading_end_time-data_loading_start_time}second")
    
    elif args.filesystem == "ramfs":
        clear_system_cache()
        clear_directory()
        prefetch_start, prefect_end, prefetch_time = prefetch_files_concurrently(selected_files, args.threads, args.ceph_mount, args.ramdisk)
        print(f"prefetching start time {prefetch_start} end {prefect_end} for total {prefetch_time}second")
        root_dir = "/mnt/ramdisk/imagenet"
        json_path = "/mnt/mycephfs/imagenet/selected_files_20GB_train.json"
        # Initialize the Dataset in 'ramfs' mode
        dataset_ramfs = ImageNetKaggle(
            root=root_dir,
            split="train",  # or "val" based on your JSON file
            ceph_mount=args.ceph_mount, 
            ramfs_mount=args.ramdisk, 
            json_=args.json_file,
            transform=train_transform
        )
        dataloader_ramfs = DataLoader(
        dataset_ramfs,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # Adjust based on your system's capabilities
        pin_memory=True
        )
        print(f"[{time.time()}] Finished creating dataset...")
        print(f"Data loading started .... ")
        data_loading_start_time = time.time()
        for _ in range (0, args.epoch):
            for batch_idx, (images, labels) in enumerate(dataloader_ramfs):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
        data_loading_end_time = time.time()
        print(f"Data loading start at {data_loading_start_time} and end {data_loading_end_time} for {data_loading_end_time-data_loading_start_time}second")
    else:
        print("Invalid filesystem option selected.")
        sys.exit(1)


if __name__ == "__main__":
    main()