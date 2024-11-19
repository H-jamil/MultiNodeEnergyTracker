import os
import time
from threading import Thread, Lock, Event, Barrier
from collections import deque
from dataclasses import dataclass
from typing import List, Deque
from datetime import datetime
import statistics

@dataclass
class EnergyMeasurement:
    timestamp: float
    cpu_core: float
    cpu_total: float

class EnergyAccumulator:
    def __init__(self):
        self.cpu_core_total = 0.0
        self.cpu_package_total = 0.0
        self.lock = Lock()
        self.last_timestamp = None

    def add(self, measurement: EnergyMeasurement):
        with self.lock:
            self.cpu_core_total += measurement.cpu_core
            self.cpu_package_total += measurement.cpu_total
            self.last_timestamp = measurement.timestamp

    def get_and_reset(self):
        with self.lock:
            totals = (self.cpu_core_total, self.cpu_package_total)
            self.cpu_core_total = 0.0
            self.cpu_package_total = 0.0
            self.last_timestamp = None
            return totals

class CircularBuffer:
    def __init__(self, maxlen: int):
        self.buffer: Deque[EnergyMeasurement] = deque(maxlen=maxlen)
        self.lock = Lock()

    def add(self, measurement: EnergyMeasurement):
        with self.lock:
            self.buffer.append(measurement)

    def get_all(self) -> List[EnergyMeasurement]:
        with self.lock:
            return list(self.buffer)

    def clear(self):
        with self.lock:
            self.buffer.clear()

class EnergyMonitor:
    def __init__(self, buffer_time_seconds: int = 2):
        self.sample_interval = 0.1  # 100ms
        self.buffer_size = int(buffer_time_seconds / self.sample_interval)
        self.buffer = CircularBuffer(self.buffer_size)
        self.accumulator = EnergyAccumulator()
        self.stop_event = Event()
        
        # Fixed 32-bit counter
        self.MAX_ENERGY_COUNTER = 2**32
        
        # Synchronization
        self.measurement_threads = 2  # Only CPU core and total
        self.sync_barrier = Barrier(self.measurement_threads)
        
        # Initialize RAPL paths
        self.rapl_dir = "/sys/class/powercap/intel-rapl"

        # Shared measurement storage
        self.current_measurement = None
        self.measurement_lock = Lock()
        
        # Last valid total energy tracking
        self.last_valid_total_energy = None
        self.total_energy_lock = Lock()
        
        # Track valid total/core ratios
        self.valid_ratios = deque(maxlen=100)  # Keep last 100 valid ratios
        self.ratio_lock = Lock()
        self.min_ratio = 1.1  # Minimum acceptable ratio

    def _read_rapl_energy(self, path: str) -> int:
        try:
            with open(path, "r") as f:
                return int(f.read())
        except (PermissionError, FileNotFoundError):
            return 0

    def _calculate_energy_delta(self, current: int, previous: int) -> float:
        if current < previous:  # Overflow occurred
            delta = (self.MAX_ENERGY_COUNTER - previous) + current
        else:
            delta = current - previous
        return delta / 1_000_000  # Convert to Joules

    def _update_ratio(self, total: float, core: float):
        """Update the running ratio of total/core energy"""
        if total > 0 and core > 0:
            ratio = total / core
            if ratio > self.min_ratio:  # Only store reasonable ratios
                with self.ratio_lock:
                    self.valid_ratios.append(ratio)

    def _get_current_ratio(self) -> float:
        """Get current average ratio, with fallback to minimum"""
        with self.ratio_lock:
            if len(self.valid_ratios) > 0:
                return statistics.median(self.valid_ratios)
            return self.min_ratio

    def _get_valid_total_energy(self, current_total: float, current_core: float) -> float:
        """Get valid total energy using dynamic ratio"""
        if current_total > 0 and current_core > 0:
            ratio = current_total / current_core
            if ratio > self.min_ratio:
                self._update_ratio(current_total, current_core)
                return current_total
        
        # If current_total is invalid, use ratio-based estimate
        return current_core * self._get_current_ratio()

    def measure_cpu_core(self):
        last_energies = {}
        
        while not self.stop_event.is_set():
            self.sync_barrier.wait()
            timestamp = time.time()
            
            core_energy = 0
            for package in os.listdir(self.rapl_dir):
                if package.startswith("intel-rapl:"):
                    core_path = os.path.join(self.rapl_dir, package, 
                                           f"{package}:0", "energy_uj")
                    energy = self._read_rapl_energy(core_path)
                    
                    if package in last_energies:
                        core_energy += self._calculate_energy_delta(
                            energy, last_energies[package])
                    last_energies[package] = energy
            
            with self.measurement_lock:
                if self.current_measurement is None:
                    self.current_measurement = EnergyMeasurement(
                        timestamp=timestamp,
                        cpu_core=core_energy,
                        cpu_total=0.0
                    )
                else:
                    self.current_measurement.cpu_core = core_energy
            
            time.sleep(self.sample_interval)

    def measure_cpu_total(self):
        last_energies = {}
        
        while not self.stop_event.is_set():
            self.sync_barrier.wait()
            timestamp = time.time()
            
            total_energy = 0
            valid_reading = False
            
            for package in os.listdir(self.rapl_dir):
                if package.startswith("intel-rapl:"):
                    package_path = os.path.join(self.rapl_dir, package, "energy_uj")
                    energy = self._read_rapl_energy(package_path)
                    
                    if package in last_energies and energy > 0:
                        delta = self._calculate_energy_delta(energy, last_energies[package])
                        if delta > 0:
                            total_energy += delta
                            valid_reading = True
                    last_energies[package] = energy
            
            with self.measurement_lock:
                if self.current_measurement is not None:
                    self.current_measurement.cpu_total = self._get_valid_total_energy(
                        total_energy, self.current_measurement.cpu_core)
                    # Add to buffer and accumulator after total energy is set
                    self.buffer.add(self.current_measurement)
                    self.accumulator.add(self.current_measurement)
                    self.current_measurement = None
            
            time.sleep(self.sample_interval)

    def print_results(self):
        while not self.stop_event.is_set():
            measurements = self.buffer.get_all()
            if measurements:
                print("\nTime                     CPU(J)    CPU_Total(J)    Ratio")
                print("-" * 60)
                for m in measurements:
                    timestamp_str = datetime.fromtimestamp(m.timestamp).strftime(
                        '%Y-%m-%d %H:%M:%S.%f')[:-3]
                    ratio = m.cpu_total / m.cpu_core if m.cpu_core > 0 else 0
                    
                    print(f"{timestamp_str}  {m.cpu_core:8.2f}  "
                          f"{m.cpu_total:12.2f}  {ratio:5.2f}")
                
                # Print accumulated energy and current ratio
                core_total, package_total = self.accumulator.get_and_reset()
                current_ratio = self._get_current_ratio()
                
                print("\nBuffer Period Totals:")
                print(f"CPU Core Energy:     {core_total:.2f} J")
                print(f"CPU Package Energy:  {package_total:.2f} J")
                print(f"Current Avg Ratio:  {current_ratio:.2f}")
                print("-" * 60)
                
                self.buffer.clear()
            time.sleep(2.0)

    def run(self):
        threads = [
            Thread(target=self.measure_cpu_core, name="CPUCoreThread"),
            Thread(target=self.measure_cpu_total, name="CPUTotalThread"),
            Thread(target=self.print_results, name="PrintThread")
        ]
        
        for t in threads:
            t.start()

        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.stop_event.set()
            for t in threads:
                t.join()

if __name__ == "__main__":
    if os.geteuid() != 0:
        print("This program needs root privileges to read RAPL energy values.")
        print("Please run with sudo.")
        exit(1)

    monitor = EnergyMonitor(buffer_time_seconds=2)
    monitor.run()