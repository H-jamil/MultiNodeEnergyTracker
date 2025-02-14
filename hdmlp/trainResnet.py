# import six
import torch
# torch._six = six
import os
import hdmlp
import hdmlp.lib.torch
import hdmlp.lib.transforms
import time
def main():
    # Set your ImageNet training directory.
    # data_dir = "/home/cc/train/ILSVRC/Data/CLS-LOC/train"
    data_dir = "/home/cc/train/subset/train"

    
    # Job configuration parameters.
    batch_size = 32       # Global batch size
    epochs = 1            # Only one epoch for testing I/O
    drop_last = False     # Do not drop the last incomplete batch

    # Define transforms:
    # 1. Decode the image.
    # 2. Resize it to 224x224, which will yield known output dimensions (224, 224, 3).
    # 3. Convert to a tensor.
    transforms_list = [
        hdmlp.lib.transforms.ImgDecode(),   # Decodes image (returns, e.g., a PIL image)
        hdmlp.lib.transforms.Resize(224, 224),# Forces output dimensions to (224, 224, 3)
        hdmlp.lib.transforms.ToTensor()       # Converts to tensor, preserving dimensions
    ]
    
    # Create an HDMLP job with the transforms.
    job = hdmlp.Job(
        data_dir,
        batch_size,
        epochs,
        'uniform',
        drop_last,
        transforms=transforms_list,
        seed=42
    )
    print("Job created")
    
    # Create the dataset.
    dataset = hdmlp.lib.torch.HDMLPImageFolder(
        data_dir,
        job
    )
    print("Dataset created")
    
    # Create the HDMLP DataLoader.
    data_loader = hdmlp.lib.torch.HDMLPDataLoader(dataset)
    print("DataLoader created")
    
    # Use CUDA if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Iterate over a few batches to test I/O.
    start_time = time.time()
    try:
        for batch_idx, (images, labels) in enumerate(data_loader):
            # Move data to GPU.
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # print(f"Batch {batch_idx}: images shape = {images.shape}, labels shape = {labels.shape}")
            if batch_idx % 5 == 0:
                print(f"Batch {batch_idx}: images shape = {images.shape}, labels shape = {labels.shape}")
        print(f"Took {time.time()-start_time} seconds")
                # break
    finally:
        job.destroy()
        print("Job destroyed")

if __name__ == "__main__":
    main()
