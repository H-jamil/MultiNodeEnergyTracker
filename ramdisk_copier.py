import os
import json
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple
import logging
from datetime import datetime
import time

class RamdiskCopier:
    def __init__(self, ramdisk_path: str, json_config: str, max_workers: int = 8):
        self.ramdisk_path = Path(ramdisk_path)
        self.json_config = json_config
        self.max_workers = max(1, min(max_workers, 8))
        self.copied_files = []
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_config(self) -> Tuple[List[str], List[str]]:
        try:
            with open(self.json_config, 'r') as f:
                config = json.load(f)
                return (config['selected_samples'], config['selected_targets'])
        except Exception as e:
            self.logger.error(f"Error loading config file: {e}")
            raise

    def copy_file(self, file_tuple: Tuple[str, str]) -> None:
        source_path, target_path = file_tuple
        try:
            source = Path(source_path)
            if not source.exists():
                self.logger.warning(f"Source file not found: {source}")
                return

            # Create relative path structure in ramdisk
            dest = self.ramdisk_path / source.relative_to(source.anchor)
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Copy the file
            shutil.copy2(source, dest)
            self.copied_files.append((str(dest), target_path))
            # self.logger.info(f"Copied: {source} -> {dest}")

        except Exception as e:
            self.logger.error(f"Error copying {source_path}: {e}")

    def save_copied_files_list(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"copied_files_{timestamp}.txt"
        
        try:
            with open(output_file, 'w') as f:
                for file_path, target_path in sorted(self.copied_files):
                    f.write(f"{file_path}\t{target_path}\n")
            self.logger.info(f"Saved copied files list to: {output_file}")
        except Exception as e:
            self.logger.error(f"Error saving copied files list: {e}")

    def run(self):
        start_time = time.time()
        self.logger.info(f"Starting copy operation with {self.max_workers} workers")
        
        # Load samples and targets from the JSON file
        samples, targets = self.load_config()
        self.logger.info(f"Loaded {len(samples)} files to copy")
        
        # Create list of tuples containing (sample_path, target_path)
        files_to_copy = list(zip(samples, targets))
        
        # Copy files using thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            executor.map(self.copy_file, files_to_copy)

        end_time = time.time()
        print(f"Data copying Phase: Start Time = {start_time}, "
        f"End Time = {end_time}, "
        f"Duration = {round((start_time - end_time),2)}s")
        
        self.save_copied_files_list()

        self.logger.info(f"Copy operation completed in {end_time - start_time:.2f} seconds")
        self.logger.info(f"Total files copied: {len(self.copied_files)}")

def main():
    parser = argparse.ArgumentParser(description='Copy files to ramdisk with multi-threading')
    parser.add_argument('ramdisk_path', help='Path to mounted ramdisk')
    parser.add_argument('json_config', help='Path to JSON config file with file list')
    parser.add_argument('--workers', type=int, default=8, help='Number of worker threads (1-8)')
    
    args = parser.parse_args()
    
    copier = RamdiskCopier(args.ramdisk_path, args.json_config, args.workers)
    copier.run()

if __name__ == "__main__":
    main()