import os
import time
import threading
import subprocess
import pynvml
from datetime import datetime

class EnergyMonitor:
    def __init__(self, sample_interval=0.1):
        self.sample_interval = sample_interval
        self.stop_event = threading.Event()
        
        # Initialize GPU monitoring
        try:
            pynvml.nvmlInit()
            self.gpu_count = pynvml.nvmlDeviceGetCount()
            if self.gpu_count > 0:
                self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.gpu_count)]
                print(f"Detected {self.gpu_count} GPU(s)")
            else:
                print("No GPUs detected")
        except Exception as e:
            print(f"GPU initialization failed: {e}")
            self.gpu_count = 0

    def measure_cpu_memory(self):
        cmd = ["perf", "stat", "-a", "-e", "power/energy-ram/", "-e", "power/energy-pkg/", "sleep", str(self.sample_interval)]
        try:
            process = subprocess.Popen(cmd, 
                                     stderr=subprocess.PIPE, 
                                     stdout=subprocess.PIPE,
                                     universal_newlines=True)
            stdout, stderr = process.communicate()
            
            memory_energy = None
            cpu_energy = None
            
            # Debug output
            print(f"Raw perf output:\n{stderr}")
            
            for line in stderr.splitlines():
                if "energy-ram" in line:
                    try:
                        memory_energy = float(line.split()[0].replace(',', ''))
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing RAM energy: {e}")
                elif "energy-pkg" in line:
                    try:
                        cpu_energy = float(line.split()[0].replace(',', ''))
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing CPU energy: {e}")
                        
            return memory_energy, cpu_energy
        except Exception as e:
            print(f"Error in perf command: {e}")
            return None, None

    def measure_gpu(self):
        if self.gpu_count == 0:
            return None
            
        total_power = 0
        try:
            for handle in self.gpu_handles:
                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    total_power += power_mw / 1000.0  # Convert to Watts
                except pynvml.NVMLError as e:
                    print(f"GPU measurement error: {e}")
                    return None
            
            return total_power * self.sample_interval  # Convert power to energy
        except Exception as e:
            print(f"Error measuring GPU: {e}")
            return None

    def run(self):
        try:
            print("\nStarting energy monitoring (Press Ctrl+C to stop)...")
            print("Timestamp               Memory Energy    CPU Energy    GPU Energy")
            print("-" * 65)
            
            while not self.stop_event.is_set():
                start_time = time.time()
                
                # Measure energies
                memory_energy, cpu_energy = self.measure_cpu_memory()
                gpu_energy = self.measure_gpu()
                
                # Format timestamp
                timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                
                # Print results
                print(f"{timestamp}    {memory_energy:>8.2f} J    {cpu_energy:>8.2f} J    {gpu_energy:>8.2f} J" if all(x is not None for x in [memory_energy, cpu_energy, gpu_energy]) else
                      f"{timestamp}    {'None':>8s}      {'None':>8s}      {'None':>8s}")
                
                # Sleep for remaining time
                elapsed = time.time() - start_time
                if elapsed < self.sample_interval:
                    time.sleep(self.sample_interval - elapsed)
                    
        except KeyboardInterrupt:
            print("\nStopping energy monitoring...")
        finally:
            if self.gpu_count > 0:
                pynvml.nvmlShutdown()

if __name__ == "__main__":
    if os.geteuid() != 0:
        print("This program needs root privileges to read energy values.")
        print("Please run with sudo.")
        exit(1)

    monitor = EnergyMonitor(sample_interval=0.1)
    monitor.run()