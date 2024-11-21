import os
import time
import threading
import subprocess
import pynvml
from datetime import datetime
from threading import Thread, Lock, Event
import argparse
import socket
import struct
import json

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
            print(f"Connected to master at {self.master_ip}:{self.control_port}/{self.data_port}")
        except Exception as e:
            print(f"Error connecting to master: {e}")
            raise

    def close(self):
        self.data_socket.close()
        self.control_socket.close()

class EnergyMonitor:
    def __init__(self, sample_interval=0.1, client=None):
        self.sample_interval = sample_interval
        self.client = client
        self.stop_event = Event()
        self.measurements = {}
        self.measurements_lock = Lock()
        self.base_time = None
        self.last_measurements = {'memory_energy': None, 'cpu_energy': None, 'gpu_energy': None}

        # Initialize GPU monitoring
        self.gpu_available = False
        try:
            pynvml.nvmlInit()
            self.gpu_count = pynvml.nvmlDeviceGetCount()
            if self.gpu_count > 0:
                self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.gpu_count)]
                self.gpu_available = True
                print(f"Detected {self.gpu_count} GPU(s)")
            else:
                print("No GPUs detected")
        except pynvml.NVMLError as e:
            print(f"NVML Initialization failed: {e}")
            self.gpu_available = False

    def run_perf_command(self):
        cmd = ["perf", "stat", "-a", "-e", "power/energy-ram/", "-e", "power/energy-pkg/", "sleep", str(self.sample_interval)]
        process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        stdout, stderr = process.communicate()
        memory_energy = 0.0
        cpu_energy = 0.0
        for line in stderr.decode().splitlines():
            line = line.strip()
            if "power/energy-ram/" in line:
                parts = line.split()
                if len(parts) >= 1:
                    try:
                        energy_value = float(parts[0].replace(',', ''))
                        memory_energy = energy_value
                    except ValueError:
                        pass
            elif "power/energy-pkg/" in line:
                parts = line.split()
                if len(parts) >= 1:
                    try:
                        energy_value = float(parts[0].replace(',', ''))
                        cpu_energy = energy_value
                    except ValueError:
                        pass
        return memory_energy, cpu_energy

    def measure_cpu_memory(self):
        next_time = self.base_time
        while not self.stop_event.is_set():
            sleep_time = next_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # We're behind schedule, proceed immediately
                pass
            memory_energy, cpu_energy = self.run_perf_command()
            with self.measurements_lock:
                if next_time not in self.measurements:
                    self.measurements[next_time] = {}
                self.measurements[next_time]['memory_energy'] = memory_energy
                self.measurements[next_time]['cpu_energy'] = cpu_energy
            next_time += self.sample_interval

    def measure_gpu(self):
        next_time = self.base_time
        while not self.stop_event.is_set():
            sleep_time = next_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                pass
            gpu_power = 0.0
            for handle in self.gpu_handles:
                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    gpu_power += power_mw / 1000.0  # Convert to Watts
                except pynvml.NVMLError:
                    pass
            gpu_energy = gpu_power * self.sample_interval
            with self.measurements_lock:
                if next_time not in self.measurements:
                    self.measurements[next_time] = {}
                self.measurements[next_time]['gpu_energy'] = gpu_energy
            next_time += self.sample_interval

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
                break
            time.sleep(5.0)

    def output_results(self):
        current_time = self.base_time
        while not self.stop_event.is_set():
            with self.measurements_lock:
                data_ready = current_time in self.measurements
                data = self.measurements.get(current_time, {})
                all_measurements_present = (
                    ('memory_energy' in data and 'cpu_energy' in data) and
                    (not self.gpu_available or 'gpu_energy' in data)
                )
                if data_ready and all_measurements_present:
                    # Update last measurements
                    for key in ['memory_energy', 'cpu_energy', 'gpu_energy']:
                        if key in data:
                            self.last_measurements[key] = data[key]
                    del self.measurements[current_time]
                    # Output measurement
                    timestamp_str = datetime.fromtimestamp(current_time).strftime('%H:%M:%S.%f')[:-3]
                    print(f"{timestamp_str}  memory: {self.last_measurements['memory_energy']} Joules  cpu: {self.last_measurements['cpu_energy']} Joules  gpu: {self.last_measurements['gpu_energy']} Joules")
                    # Send data to master
                    if self.client:
                        data_to_send = {
                            'node_id': self.client.node_id,
                            'timestamp': current_time,
                            'memory_energy': self.last_measurements['memory_energy'],
                            'cpu_energy': self.last_measurements['cpu_energy'],
                            'gpu_energy': self.last_measurements['gpu_energy']
                        }
                        json_data = json.dumps(data_to_send)
                        msg_length = struct.pack('!I', len(json_data))
                        try:
                            self.client.data_socket.sendall(msg_length)
                            self.client.data_socket.sendall(json_data.encode())
                            print(f"Sent data to master: {timestamp_str}")
                        except Exception as e:
                            print(f"Error sending data: {e}")
                    current_time += self.sample_interval
                elif data_ready and not all_measurements_present:
                    # Measurements are incomplete, wait
                    pass
                else:
                    # No data yet, check if we should fill missing data
                    if current_time + self.sample_interval < time.time():
                        # Fill missing data with last known values
                        timestamp_str = datetime.fromtimestamp(current_time).strftime('%H:%M:%S.%f')[:-3]
                        print(f"{timestamp_str}  memory: {self.last_measurements['memory_energy']} Joules  cpu: {self.last_measurements['cpu_energy']} Joules  gpu: {self.last_measurements['gpu_energy']} Joules (Filled)")
                        # Send data to master
                        if self.client:
                            data_to_send = {
                                'node_id': self.client.node_id,
                                'timestamp': current_time,
                                'memory_energy': self.last_measurements['memory_energy'],
                                'cpu_energy': self.last_measurements['cpu_energy'],
                                'gpu_energy': self.last_measurements['gpu_energy']
                            }
                            json_data = json.dumps(data_to_send)
                            msg_length = struct.pack('!I', len(json_data))
                            try:
                                self.client.data_socket.sendall(msg_length)
                                self.client.data_socket.sendall(json_data.encode())
                                print(f"Sent data to master: {timestamp_str}")
                            except Exception as e:
                                print(f"Error sending data: {e}")
                        current_time += self.sample_interval
            time.sleep(0.01)

    def run(self):
        self.base_time = time.time()
        self.base_time = self.base_time - (self.base_time % self.sample_interval)
        threads = []
        cpu_memory_thread = Thread(target=self.measure_cpu_memory)
        threads.append(cpu_memory_thread)
        if self.gpu_available:
            gpu_thread = Thread(target=self.measure_gpu)
            threads.append(gpu_thread)
        heartbeat_thread = Thread(target=self.send_heartbeat)
        threads.append(heartbeat_thread)
        output_thread = Thread(target=self.output_results)
        threads.append(output_thread)
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
        print("This program needs root privileges to read energy values.")
        print("Please run with sudo.")
        exit(1)

    parser = argparse.ArgumentParser(description="Energy Monitoring Tool")
    parser.add_argument('--sampling_time', type=float, default=0.1, help='Sampling time in seconds')
    parser.add_argument('--node_id', type=str, default='compute1', help='Node ID')
    parser.add_argument('--master_ip', type=str, default='10.52.3.53', help='Master IP address')
    parser.add_argument('--control_port', type=int, default=5000, help='Control port')
    parser.add_argument('--data_port', type=int, default=5001, help='Data port')
    args = parser.parse_args()

    # Create and connect client
    client = EnergyClient(
        master_ip=args.master_ip,
        control_port=args.control_port,
        data_port=args.data_port,
        node_id=args.node_id
    )
    try:
        client.connect()
        print(f"Client successfully connected to master")
    except Exception as e:
        print(f"Failed to connect to master: {e}")
        exit(1)

    # Create and run monitor
    monitor = EnergyMonitor(sample_interval=args.sampling_time, client=client)
    monitor.run()
