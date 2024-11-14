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
        try:
            self.control_socket.connect((self.master_ip, self.control_port))
            self.data_socket.connect((self.master_ip, self.data_port))
            # Removed sending node identification
            print(f"Connected to master at {self.master_ip}:{self.control_port}/{self.data_port}")
        except Exception as e:
            print(f"Error connecting to master: {e}")
            raise
    

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
        self.measurement_threads = 3  # Adjusted later if GPU monitoring is disabled
        self.sync_barrier = None  # Will be initialized after determining thread count
        
        # Initialize RAPL paths
        self.rapl_dir = "/sys/class/powercap/intel-rapl"
        
        # Initialize NVIDIA GPU monitoring
        self.gpu_available = False
        try:
            pynvml.nvmlInit()
            self.gpu_count = pynvml.nvmlDeviceGetCount()
            if self.gpu_count > 0:
                self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) 
                                   for i in range(self.gpu_count)]
                self.gpu_available = True
                print(f"Detected {self.gpu_count} GPU(s)")
            else:
                print("No GPUs detected")
        except pynvml.NVMLError as e:
            print(f"NVML Initialization failed: {e}")
            self.gpu_available = False
        
        # Adjust the number of measurement threads based on GPU availability
        self.measurement_threads = 2 + (1 if self.gpu_available else 0)
        self.sync_barrier = Barrier(self.measurement_threads)
        
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
                    # Add measurement to buffer and accumulator
                    self.buffer.add(self.current_measurement)
                    self.accumulator.add(self.current_measurement)
                    self.current_measurement = None
            
            time.sleep(self.sample_interval)

    def measure_no_gpu(self):
        while not self.stop_event.is_set():
            self.sync_barrier.wait()
            # No GPUs to measure, but we need to synchronize and proceed
            with self.measurement_lock:
                if self.current_measurement is not None:
                    self.current_measurement.gpu_readings = []
                    # Add measurement to buffer and accumulator
                    self.buffer.add(self.current_measurement)
                    self.accumulator.add(self.current_measurement)
                    self.current_measurement = None
            time.sleep(self.sample_interval)

    def push_results(self):
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
                    self.client.data_socket.sendall(msg_length)
                    self.client.data_socket.sendall(json_data.encode())
                    print(f"Sent data to master: {len(measurements)} measurements")
                except Exception as e:
                    print(f"Error sending data: {e}")
                
                self.buffer.clear()
            time.sleep(2.0)  # Adjust the push interval as needed
    def send_heartbeat(self):
        while not self.stop_event.is_set():
            heartbeat = {
                'type': 'heartbeat',
                'node_id': self.client.node_id,
                'timestamp': time.time(),
                'status': 'active'
            }
            try:
                json_data = json.dumps(heartbeat)
                msg_length = struct.pack('!I', len(json_data))
                self.client.control_socket.sendall(msg_length)
                self.client.control_socket.sendall(json_data.encode())
                print(f"Heartbeat sent at {heartbeat['timestamp']}")
            except Exception as e:
                print(f"Error sending heartbeat: {e}")
                break  # Exit the loop if the socket is broken
            time.sleep(5.0)  # Heartbeat interval

    def run(self):
        threads = [
            Thread(target=self.measure_cpu_core, name="CPUCoreThread"),
            Thread(target=self.measure_cpu_total, name="CPUTotalThread"),
            Thread(target=self.push_results, name="PushResultsThread"),
            Thread(target=self.send_heartbeat, name="HeartbeatThread")
        ]
        
        # Start GPU measurement thread if GPUs are available
        if self.gpu_available:
            threads.append(Thread(target=self.measure_gpu, name="GPUThread"))
        else:
            threads.append(Thread(target=self.measure_no_gpu, name="NoGPUThread"))
        
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
            if self.gpu_available:
                pynvml.nvmlShutdown()

if __name__ == "__main__":
    if os.geteuid() != 0:
        print("This program needs root privileges to read RAPL energy values.")
        print("Please run with sudo.")
        exit(1)

    # Node configuration
    NODE_ID = "node1"  # Change this for each node
    MASTER_IP = "localhost"  # Change this to your master's IP
    CONTROL_PORT = 5000
    DATA_PORT = 5001

    # Create and connect client
    client = EnergyClient(
        master_ip=MASTER_IP,
        control_port=CONTROL_PORT,
        data_port=DATA_PORT,
        node_id=NODE_ID
    )
    try:
        client.connect()
        print(f"Client successfully connected to master")
    except Exception as e:
        print(f"Failed to connect to master: {e}")
        exit(1)

    # Create and run monitor
    monitor = EnergyMonitor(buffer_time_seconds=2, client=client)
    monitor.run()
