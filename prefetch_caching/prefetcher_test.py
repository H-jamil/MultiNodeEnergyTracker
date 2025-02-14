import os
import time
import threading
from collections import defaultdict, OrderedDict
from queue import Queue, Empty
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import argparse
# 1. Plan File Parser with Error Handling
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
                    current_epoch = int(parts[1].split()[-1])
                    current_step = int(parts[2].split()[1].strip(':'))
                else:
                    if current_epoch is not None and current_step is not None:
                        plan[current_epoch][current_step].append(line)
    except Exception as e:
        print(f"Error parsing plan file: {str(e)}")
    
    return dict(plan)

# 2. Cache Manager with Safe Tensor Handling
class TrainingCache:
    def __init__(self, max_size_gb=20):
        self.max_size = max_size_gb * 1024**3
        self.current_size = 0
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def add(self, key: str, data: torch.Tensor):
        with self.lock:
            if data is None:
                return
                
            size = data.element_size() * data.nelement()
            while self.current_size + size > self.max_size and self.cache:
                self._evict()
                
            self.cache[key] = (data, size)
            self.current_size += size
            self.cache.move_to_end(key)

    def get(self, key: str) -> torch.Tensor:
        with self.lock:
            return self.cache.get(key, (None, 0))[0]  # Return tensor or None

    def _evict(self):
        oldest = next(iter(self.cache))
        self.current_size -= self.cache[oldest][1]
        del self.cache[oldest]

# 3. Robust Prefetcher with Error Handling
class PlanAwarePrefetcher:
    def __init__(self, plan: dict, cache: TrainingCache, transform):
        self.plan = plan
        self.cache = cache
        self.transform = transform
        self.prefetch_queue = Queue()
        self.workers = []
        self.active = True
        self.worker_count = 2
        self.metrics = {'load_times': []}
        
        self.control_thread = threading.Thread(target=self._adjust_workers)
        self.control_thread.start()
        self._scale_workers(self.worker_count)

    def schedule_prefetch(self, epoch: int, step: int):
        lookahead = self._calculate_lookahead()
        for offset in range(1, lookahead + 1):
            next_step = step + offset
            if next_step in self.plan.get(epoch, {}):
                for path in self.plan[epoch][next_step]:
                    if self.cache.get(path) is None:  # Corrected check
                        self.prefetch_queue.put(path)

    def _calculate_lookahead(self):
        if not self.metrics['load_times']:
            return 1
        avg_load_time = sum(self.metrics['load_times'][-10:])/10
        return max(1, min(5, int(0.2 / (avg_load_time + 1e-6))))

    def _prefetch_worker(self):
        while True:
            try:
                path = self.prefetch_queue.get(timeout=1)
                if path is None:
                    break
                
                try:
                    start = time.time()
                    img = Image.open(path).convert('RGB')
                    tensor = self.transform(img)
                    self.cache.add(path, tensor)
                    self.metrics['load_times'].append(time.time() - start)
                except Exception as e:
                    print(f"Prefetch error {path}: {str(e)}")
                
            except Empty:
                continue

    def _adjust_workers(self):
        while self.active:
            time.sleep(5)
            if len(self.metrics['load_times']) < 10:
                continue
                
            avg_load = sum(self.metrics['load_times'][-10:])/10
            qsize = self.prefetch_queue.qsize()
            
            new_count = self.worker_count
            if avg_load > 0.1 and qsize > 10:
                new_count += 1
            elif avg_load < 0.05 or qsize < 5:
                new_count = max(1, new_count - 1)
            
            if new_count != self.worker_count:
                self._scale_workers(new_count)

    def _scale_workers(self, new_count):
        while len(self.workers) < new_count:
            worker = threading.Thread(target=self._prefetch_worker)
            worker.start()
            self.workers.append(worker)
        
        while len(self.workers) > new_count:
            self.prefetch_queue.put(None)
            self.workers.pop().join()

    def stop(self):
        self.active = False
        self._scale_workers(0)
        self.control_thread.join()

# 4. Safe Dataset Class
class PlanDataset(Dataset):
    def __init__(self, plan_path, cache, transform):
        self.plan = parse_plan_file(plan_path)
        self.cache = cache
        self.transform = transform
        self.prefetcher = None
        self.current_epoch = 0
        
        # Create step list with validation
        self.steps = []
        for epoch in sorted(self.plan.keys()):
            for step in sorted(self.plan[epoch].keys()):
                if self.plan[epoch][step]:
                    self.steps.append((epoch, step))

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, idx):
        try:
            epoch, step = self.steps[idx]
            paths = self.plan[epoch][step]
            
            if not self.prefetcher:
                self.prefetcher = PlanAwarePrefetcher(self.plan, self.cache, self.transform)
            
            self.prefetcher.schedule_prefetch(epoch, step)
            
            batch = []
            for path in paths:
                tensor = self.cache.get(path)
                if tensor is None:
                    try:
                        img = Image.open(path).convert('RGB')
                        tensor = self.transform(img)
                    except Exception as e:
                        print(f"Error loading {path}: {str(e)}")
                        tensor = torch.zeros(3, 224, 224)
                batch.append(tensor)
            
            return torch.stack(batch), torch.zeros(len(batch))
        except Exception as e:
            print(f"Dataset error: {str(e)}")
            return torch.zeros(0), torch.zeros(0)

# 5. DDP Training with Safe Initialization
def setup(rank, world_size):
    try:
        device_id = rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            rank=rank,
            world_size=world_size
        )
        return device_id
    except Exception as e:
        print(f"Distributed setup failed: {str(e)}")
        raise

def main(rank, world_size):
    try:
        device_id = setup(rank, world_size)
        print(f"Initialized rank {rank} on device {device_id}")
        
        # Training parameters
        batch_size = 32
        num_epochs = 1
        cache = TrainingCache(max_size_gb=20)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Dataset with validation
        dataset = PlanDataset(
            f"plan_rank{rank}_1.txt",
            cache,
            transform
        )
        
        # DataLoader with safe collation
        def collate_fn(batch):
            batch = [item for item in batch if item[0].shape[0] > 0]
            if not batch:
                return torch.zeros(0), torch.zeros(0)
            return torch.utils.data.default_collate(batch)
        
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn
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
        
        # Training loop with validation
        start_time = time.time()
        for epoch in range(num_epochs):
            dataset.current_epoch = epoch
            model.train()
            
            for batch_idx, (data, _) in enumerate(loader):
                if data.nelement() == 0:
                    continue
                
                data = data.squeeze(0).to(device_id, non_blocking=True)
                
                try:
                    optimizer.zero_grad()
                    outputs = model(data)
                    loss = outputs.mean()
                    loss.backward()
                    optimizer.step()
                    
                    # if rank == 0:
                    print(f"Epoch {epoch} Step {batch_idx} Loss: {loss.item()}")
                        
                except Exception as e:
                    print(f"Training error: {str(e)}")
                if batch_idx == 10:
                    print(f"Time taken for training is {time.time()-start_time} seconds")
                    break
        if rank == 0:
            torch.save(model.state_dict(), "model.pth")
            
    except Exception as e:
        print(f"Main execution failed: {str(e)}")
    finally:
        if 'dataset' in locals() and dataset.prefetcher:
            dataset.prefetcher.stop()
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world_size', type=int, default=2)
    args = parser.parse_args()
    
    main(args.rank, args.world_size)