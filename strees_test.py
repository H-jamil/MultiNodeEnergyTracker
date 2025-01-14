#!/usr/bin/env python3

import time
import argparse

import numpy as np

# Import torch only if installed (for GPU usage).
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def max_usage(n_seconds=10):
    # ---------------------------
    # 1. Set up large CPU array
    # ---------------------------
    # For example, a 4000x4000 array = 16 million elements.
    # Each double ~8 bytes => ~128 MB memory.
    # Adjust size if your system has less memory or to stress more.
    cpu_array = np.random.randn(4000, 4000)

    # ---------------------------
    # 2. Set up large GPU array (if PyTorch + CUDA available)
    # ---------------------------
    if HAS_TORCH and torch.cuda.is_available():
        device = torch.device("cuda")
        # For example, a 2000x2000 float32 matrix => 4 million elements => ~16 MB.
        gpu_tensor = torch.randn(2000, 2000, dtype=torch.float32, device=device)
        print("GPU detected: using PyTorch to stress GPU.")
    else:
        gpu_tensor = None
        print("No GPU or PyTorch not installed: skipping GPU usage.")

    # -----------------------------------
    # 3. Keep multiplying until time's up
    # -----------------------------------
    end_time = time.time() + n_seconds
    iteration = 0

    print(f"Starting stress for {n_seconds} seconds...")
    while time.time() < end_time:
        # Heavy CPU operation (matrix multiplication)
        _ = cpu_array @ cpu_array

        # Heavy GPU operation (matrix multiplication)
        if gpu_tensor is not None:
            _ = gpu_tensor @ gpu_tensor

        iteration += 1
    
    print(f"Completed {iteration} multiplications in ~{n_seconds} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stress CPU, memory, and GPU for N seconds.")
    parser.add_argument("--seconds", type=int, default=10,
                        help="Number of seconds to run the usage test.")
    args = parser.parse_args()
    print(f"start time {time.time()}")
    max_usage(args.seconds)
    print(f"end time {time.time()}")
