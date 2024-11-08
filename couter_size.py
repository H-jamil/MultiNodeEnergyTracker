import os
import time
import pynvml
from threading import Thread, Lock, Event, Barrier
from collections import deque
from dataclasses import dataclass
from typing import List, Deque
from datetime import datetime

def check_counter_size():
    """Check RAPL counter size by monitoring wraparound"""
    rapl_path = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
    
    try:
        with open(rapl_path, "r") as f:
            start_val = int(f.read())
        
        # Monitor until wraparound occurs
        prev_val = start_val
        while True:
            time.sleep(0.1)
            with open(rapl_path, "r") as f:
                curr_val = int(f.read())
            
            if curr_val < prev_val:
                # Wraparound occurred
                if curr_val < 2**32:
                    print("32-bit counter detected")
                    return 32
                else:
                    print("64-bit counter detected")
                    return 64
            prev_val = curr_val
            
    except Exception as e:
        print(f"Error checking counter size: {e}")
        return 64  # Default to 64-bit for safety

def main():
    check_counter_size()

if __name__ == "__main__":
    main()