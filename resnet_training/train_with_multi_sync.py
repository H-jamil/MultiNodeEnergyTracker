import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import ImageFolder

###############################################################################
# 1. Distributed Setup and Cleanup
###############################################################################
def setup_distributed():
    """
    Initialize distributed training environment using NCCL.
    Expects environment vars: WORLD_SIZE, RANK, MASTER_ADDR, MASTER_PORT.
    """
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    # One GPU per rank assumption
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda", rank % torch.cuda.device_count())
    return device, rank, world_size

def cleanup_distributed():
    """Destroy the default process group."""
    dist.destroy_process_group()

###############################################################################
# 2. Simple ResNet50 System
###############################################################################
class ResNet50System(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained=False)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

###############################################################################
# 3. Dataset / DataLoader
###############################################################################
def get_dataset(data_path):
    """
    Return an ImageNet-like dataset using ImageFolder.
    Update `data_path` for your environment.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return ImageFolder(root=data_path, transform=transform)

###############################################################################
# 4. Main: Two Steps
#    Step 1: Collect 100 local losses (no DDP sync), store to disk.
#    Step 2: Load those 100 losses, do a "dummy backward" on each
#            which DOES sync with DDP.
###############################################################################
def main():
    # -------------------------------------------------------------------------
    # A) Initialize Distributed
    # -------------------------------------------------------------------------
    device, rank, world_size = setup_distributed()
    print(f"[Rank {rank}] Running on device={device}, world_size={world_size}")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # -------------------------------------------------------------------------
    # B) Build Dataset / DataLoader
    #
    # Make sure your dataset has at least 100 batches *per rank*,
    # otherwise you'll hit StopIteration before collecting 100 losses.
    # -------------------------------------------------------------------------
    data_path = '/home/cc/imagenet/ILSVRC/Data/CLS-LOC/train'  # Update as needed
    dataset = get_dataset(data_path)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )

    # -------------------------------------------------------------------------
    # C) Model & Optimizer
    # -------------------------------------------------------------------------
    model = ResNet50System().to(device)
    model = DDP(model, device_ids=[device], output_device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    MAX_ITERS = 100  # Number of local iterations
    local_losses = []
    output_file = f"rank_{rank}_losses.pt"

    # -------------------------------------------------------------------------
    # STEP 1: Get 100 local losses, no DDP gradient sync
    #
    # - We call `with model.no_sync():` so that DDP doesn't do all-reduce
    #   for gradients in Step 1. Only local updates per rank.
    # - We store the final scalar loss each time in local_losses[].
    # -------------------------------------------------------------------------
    print(f"[Rank {rank}] STEP 1: Gathering 100 local losses (DDP sync disabled).")
    data_iter = iter(dataloader)

    for i in range(MAX_ITERS):
        try:
            # -- Data Load
            load_start = time.time()
            images, labels = next(data_iter)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            torch.cuda.synchronize()
            load_end = time.time()

            # -- Forward & Backward (skip DDP sync)
            # with model.no_sync():
            compute_start = time.time()
            outputs = model(images)
            loss = model.module.criterion(outputs, labels)
            optimizer.zero_grad()
            compute_end = time.time()

            
            compute_bs_start = time.time()
            loss.backward()   # No gradient sync
            optimizer.step()
            torch.cuda.synchronize()
            compute_bs_end = time.time()

            local_losses.append(loss.detach().cpu().item())

            # if rank == 0:
            print(f"Step1 Iter={i:02d} "
                    f"| DataLoad={load_end - load_start:.4f}s "
                    f"| Compute={compute_end - compute_start:.4f}s "
                    f"| backword+sync={compute_bs_end - compute_bs_start:.4f}s "
                    f"| LocalLoss={loss.item():.4f}")

        except StopIteration:
            print(f"[Rank {rank}] Dataloader exhausted before {MAX_ITERS} iterations!")
            break

    # Save the 100 local losses to disk (one file per rank)
    # torch.save(local_losses, output_file)
    # print(f"[Rank {rank}] Saved {len(local_losses)} losses to {output_file}.")

    # Ensure all ranks finish step 1
    dist.barrier()

    # -------------------------------------------------------------------------
    # STEP 2: Load those 100 losses from disk, do a "dummy backward" that
    #         DOES sync across ranks (DDP). 
    #
    # The trick: each loaded scalar has no gradient connection to the model
    # by default, so we build a small graph that depends on the model params.
    # Then .backward() triggers an all-reduce.
    # -------------------------------------------------------------------------
    # print(f"[Rank {rank}] STEP 2: Training with loaded losses (DDP sync enabled).")

    # loaded_losses = torch.load(output_file)  # list of floats from Step 1
    # for i, scalar_loss_value in enumerate(loaded_losses):
    #     # Convert to GPU float
    #     loaded_loss_gpu = torch.tensor(scalar_loss_value, device=device)

    #     # Build a small graph that depends on the model parameters.
    #     # For example, sum up all parameters => param_sum
    #     param_sum = torch.zeros(1, device=device)
    #     for p in model.parameters():
    #         param_sum = param_sum + p.sum()

    #     # Multiply the loaded scalar by param_sum
    #     # => This "dummy_loss" depends on the model parameters
    #     dummy_loss = loaded_loss_gpu * param_sum

    #     optimizer.zero_grad()
    #     # This backward call triggers DDP's gradient sync
    #     dummy_loss.backward()
    #     optimizer.step()
    #     torch.cuda.synchronize()

    #     # Barrier to ensure all ranks finish iteration i in lockstep
    #     dist.barrier()

    #     # if rank == 0:
    #     print(f"Step2 Iter={i:02d} | DummyLoss={dummy_loss.item():.4f}")

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    cleanup_distributed()
    print(f"[Rank {rank}] Done.")

if __name__ == "__main__":
    main()
