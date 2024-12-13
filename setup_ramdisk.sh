#!/bin/bash
# Create a 21GB ramdisk
RAMDISK_SIZE_MB=$((21 * 1024))  # Convert GB to MB
MOUNT_POINT="/mnt/ramdisk"

# Create mount point if it doesn't exist
sudo mkdir -p $MOUNT_POINT

# Mount ramdisk
sudo mount -t tmpfs -o size=${RAMDISK_SIZE_MB}M tmpfs $MOUNT_POINT

# Set permissions
sudo chmod 777 $MOUNT_POINT

# Run the Python script
# python ramdisk_copier.py $MOUNT_POINT config.json --workers 8

# Note: To unmount later, use:
# sudo umount $MOUNT_POINT