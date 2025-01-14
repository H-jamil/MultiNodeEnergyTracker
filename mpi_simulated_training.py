#!/usr/bin/env python

import sys
import time

import torch
import torch.nn as nn
import torchvision
import numpy as np
from mpi4py import MPI

###############################################################################
# 1. Simple ResNet50 Model
###############################################################################
class ResNet50System(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained=False)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

###############################################################################
# 2. Flatten / Unflatten Utilities
###############################################################################
def flatten_model_params(model: nn.Module) -> torch.Tensor:
    param_list = []
    for p in model.parameters():
        param_list.append(p.detach().cpu().view(-1).float())
    flat_torch = torch.cat(param_list, dim=0)
    return flat_torch

def unflatten_model_params(model: nn.Module, flat_torch: torch.Tensor):
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        slice_ = flat_torch[idx : idx+numel].view_as(p.data)
        p.data.copy_(slice_)
        idx += numel

###############################################################################
# 3. Baseline All-Reduce (Gather-Then-Broadcast)
###############################################################################
def baseline_allreduce(flat_tensor: torch.Tensor, comm: MPI.Intracomm):
    """
    Very basic "gather-then-broadcast" all-reduce:
      1) Rank 0 gathers data from all ranks
      2) Rank 0 sums
      3) Rank 0 broadcasts
      4) Everyone divides by size => average
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_array = flat_tensor.numpy()  # shape = (num_params,)
    local_len = local_array.size

    # Rank 0 will gather everything
    if rank == 0:
        gather_buf = np.zeros(size * local_len, dtype=np.float32)
    else:
        gather_buf = None

    comm.Gather(local_array, gather_buf, root=0)

    if rank == 0:
        gather_buf = gather_buf.reshape(size, local_len)
        # sum over dimension=0 (the rank dimension)
        reduced_sum = np.sum(gather_buf, axis=0)  # shape=(local_len,)
    else:
        reduced_sum = np.zeros(local_len, dtype=np.float32)

    # Broadcast the sum
    comm.Bcast(reduced_sum, root=0)

    # Divide by size => average
    reduced_sum /= size

    # Overwrite local array
    local_array[:] = reduced_sum[:]
    # Copy back to flat_tensor
    flat_tensor.copy_(torch.from_numpy(local_array))

###############################################################################
# 4. Custom Ring All-Reduce
###############################################################################
def ring_allreduce(flat_tensor: torch.Tensor, comm: MPI.Intracomm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    flat_array = flat_tensor.numpy()
    local_size = flat_array.size
    left = (rank - 1) % size
    right = (rank + 1) % size

    send_buf = flat_array.copy()
    recv_buf = np.zeros(local_size, dtype=np.float32)

    for step in range(size - 1):
        req = comm.Isend(send_buf, dest=right, tag=step)
        comm.Recv(recv_buf, source=left, tag=step)
        req.Wait()
        flat_array[:] = flat_array + recv_buf[:]
        send_buf[:] = flat_array[:]

    flat_array[:] = flat_array / size
    flat_tensor.copy_(torch.from_numpy(flat_array))

###############################################################################
# 5. Built-In MPI Allreduce
###############################################################################
def builtin_allreduce(flat_tensor: torch.Tensor, comm: MPI.Intracomm):
    flat_array = flat_tensor.numpy()
    comm.Allreduce(MPI.IN_PLACE, flat_array, op=MPI.SUM)
    flat_array[:] = flat_array / comm.Get_size()
    flat_tensor.copy_(torch.from_numpy(flat_array))

###############################################################################
# 6. Main
###############################################################################
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Parse method from command line
    # e.g. python mpi_random_init.py baseline
    #      python mpi_random_init.py ring
    #      python mpi_random_init.py allreduce
    if len(sys.argv) > 1:
        method = sys.argv[1].strip().lower()
    else:
        method = "baseline"  # default

    model = ResNet50System()

    local_file = f"rank_{rank}_init.pth"
    torch.save(model.state_dict(), local_file)
    print(f"[Rank {rank}] Saved local model {local_file} (random init).")

    NUM_SYNCS = 10
    print(f"[Rank {rank} start time {time.time()}]")
    for i in range(NUM_SYNCS):
        flat_tensor = flatten_model_params(model)

        start = time.time()
        if method == "ring":
            ring_allreduce(flat_tensor, comm)
        elif method == "allreduce":
            builtin_allreduce(flat_tensor, comm)
        else:
            baseline_allreduce(flat_tensor, comm)
        end = time.time()

        unflatten_model_params(model, flat_tensor)
        comm.Barrier()

        if rank == 0 and i % 10 == 0:
            print(f"[Iter {i:02d}] {method.upper()} allreduce done in {end - start:.5f}s")

    comm.Barrier()
    print(f"[Rank {rank}] Completed {NUM_SYNCS} {method} allreduce iterations.")
    print(f"[Rank {rank} end time {time.time()}]")
    if rank == 0:
        print("[All ranks] Done.")

if __name__ == "__main__":
    main()
