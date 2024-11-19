import os
import time
import pynvml
from threading import Thread, Lock, Event, Barrier
from collections import deque
from dataclasses import dataclass
from typing import List, Deque
from datetime import datetime
import statistics
import json
import socket
import struct

@dataclass
class EnergyMeasurement:
    timestamp: float
    cpu_core: float
    cpu_total: float
    gpu_readings: List[float]

class EnergyClient:
    def __init__(self, master_ip: str, control_port: int, data_port: int, node_id: str):
        self.master_ip = master_ip
        self.control_port = control_port
        self.data_port = data_port
        self.node_id = node_id
        self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
    def connect(self):
        self.control_socket.connect((self.master_ip, self.control_port))
        self.data_socket.connect((self.master_ip, self.data_port))
        # Send node identification
        self.control_socket.send(self.node_id.encode())

    def close(self):
        self.data_socket.close()
        self.control_socket.close()

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

class EnergyAccumulator:
    def __init__(self):
        self.cpu_core_total = 0.0
        self.cpu_package_total = 0.0
        self.gpu_energy_total = 0.0
        self.lock = Lock()
        self.last_timestamp = None

    def add(self, measurement: EnergyMeasurement):
        with self.lock:
            self.cpu_core_total += measurement.cpu_core
            self.cpu_package_total += measurement.cpu_total
            
            if self.last_timestamp is not None:
                duration = measurement.timestamp - self.last_timestamp
                for gpu_power in measurement.gpu_readings:
                    self.gpu_energy_total += gpu_power * duration
            
            self.last_timestamp = measurement.timestamp

    def get_and_reset(self):
        with self.lock:
            totals = (self.cpu_core_total, self.cpu_package_total, self.gpu_energy_total)
            self.cpu_core_total = 0.0
            self.cpu_package_total = 0.0
            self.gpu_energy_total = 0.0
            self.last_timestamp = None
            return totals

class EnergyMonitor:
    def __init__(self, buffer_time_seconds: int = 2, client: EnergyClient = None):
        self.sample_interval = 0.1  # 100ms
        self.buffer_size = int(buffer_time_seconds / self.sample_interval)
        self.buffer = CircularBuffer(self.buffer_size)
        self.accumulator = EnergyAccumulator()
        self.stop_event = Event()
        self.client = client
        
        # Fixed 32-bit counter
        self.MAX_ENERGY_COUNTER = 2**32
        
        # Synchronization
        self.measurement_threads = 3
        self.sync_barrier = Barrier(self.measurement_threads)
        
        # Initialize RAPL paths
        self.rapl_dir = "/sys/class/powercap/intel-rapl"
        
        # Initialize NVIDIA GPU monitoring
        pynvml.nvmlInit()
        self.gpu_count = pynvml.nvmlDeviceGetCount()
        self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) 
                           for i in range(self.gpu_count)]

        # Shared measurement storage
        self.current_measurement = None
        self.measurement_lock = Lock()
        
        # Track valid total/core ratios
        self.valid_ratios = deque(maxlen=100)
        self.ratio_lock = Lock()
        self.min_ratio = 1.1

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
        if total > 0 and core > 0:
            ratio = total / core
            if ratio > self.min_ratio:
                with self.ratio_lock:
                    self.valid_ratios.append(ratio)

    def _get_current_ratio(self) -> float:
        with self.ratio_lock:
            if len(self.valid_ratios) > 0:
                return statistics.median(self.valid_ratios)
            return self.min_ratio

    def _get_valid_total_energy(self, current_total: float, current_core: float) -> float:
        if current_total > 0 and current_core > 0:
            ratio = current_total / current_core
            if ratio > self.min_ratio:
                self._update_ratio(current_total, current_core)
                return current_total
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
                        cpu_total=0.0,
                        gpu_readings=[]
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
            for package in os.listdir(self.rapl_dir):
                if package.startswith("intel-rapl:"):
                    package_path = os.path.join(self.rapl_dir, package, "energy_uj")
                    energy = self._read_rapl_energy(package_path)
                    
                    if package in last_energies and energy > 0:
                        delta = self._calculate_energy_delta(energy, last_energies[package])
                        if delta > 0:
                            total_energy += delta
                    last_energies[package] = energy
            
            with self.measurement_lock:
                if self.current_measurement is not None:
                    self.current_measurement.cpu_total = self._get_valid_total_energy(
                        total_energy, self.current_measurement.cpu_core)
            
            time.sleep(self.sample_interval)

    def measure_gpu(self):
        while not self.stop_event.is_set():
            self.sync_barrier.wait()
            timestamp = time.time()
            
            gpu_readings = []
            for handle in self.gpu_handles:
                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    gpu_readings.append(power_mw / 1000.0)  # Convert to Watts
                except pynvml.NVMLError:
                    gpu_readings.append(0.0)
            
            with self.measurement_lock:
                if self.current_measurement is not None:
                    self.current_measurement.gpu_readings = gpu_readings
                    self.buffer.add(self.current_measurement)
                    self.accumulator.add(self.current_measurement)
                    self.current_measurement = None
            
            time.sleep(self.sample_interval)

    def print_results(self):
        while not self.stop_event.is_set():
            measurements = self.buffer.get_all()
            if measurements and self.client:
                # Convert measurements to JSON
                data = {
                    'node_id': self.client.node_id,
                    'measurements': [
                        {
                            'timestamp': m.timestamp,
                            'cpu_core': m.cpu_core,
                            'cpu_total': m.cpu_total,
                            'gpu_readings': m.gpu_readings
                        } for m in measurements
                    ]
                }
                
                # Send data to master
                json_data = json.dumps(data)
                msg_length = struct.pack('!I', len(json_data))
                try:
                    self.client.data_socket.send(msg_length)
                    self.client.data_socket.send(json_data.encode())
                except Exception as e:
                    print(f"Error sending data: {e}")
                
                self.buffer.clear()
            time.sleep(2.0)

    def run(self):
        threads = [
            Thread(target=self.measure_cpu_core, name="CPUCoreThread"),
            Thread(target=self.measure_cpu_total, name="CPUTotalThread"),
            Thread(target=self.measure_gpu, name="GPUThread"),
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
            if self.client:
                self.client.close()
            pynvml.nvmlShutdown()

if __name__ == "__main__":
    if os.geteuid() != 0:
        print("This program needs root privileges to read RAPL energy values.")
        print("Please run with sudo.")
        exit(1)

    # Create and connect client
    client = EnergyClient(
        master_ip="localhost",  # Change this to your master's IP
        control_port=5000,
        data_port=5001,
        node_id="node1"  # Change this for each client
    )
    client.connect()
    print(f"Client successfully connected to controller")
    # Create and run monitor
    monitor = EnergyMonitor(buffer_time_seconds=2, client=client)
    monitor.run()