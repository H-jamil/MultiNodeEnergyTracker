import os
import time
import threading
import subprocess
import pynvml
from datetime import datetime
from threading import Thread, Event
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
        self.last_measurements = {'memory_energy': 0.0, 'cpu_energy': 0.0, 'gpu_energy': 0.0}

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
        try:
            process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                print(f"Perf command failed with return code {process.returncode}")
                print(f"stderr: {stderr}")
                print(f"stdout: {stdout}")
                return None, None
        except Exception as e:
            print(f"Exception running perf command: {e}")
            return None, None

        memory_energy = 0.0
        cpu_energy = 0.0
        for line in stderr.splitlines():
            line = line.strip()
            if "power/energy-ram/" in line:
                try:
                    memory_energy = float(line.split()[0].replace(',', ''))
                except (ValueError, IndexError) as e:
                    print(f"Error parsing RAM energy: {e}")
            elif "power/energy-pkg/" in line:
                try:
                    cpu_energy = float(line.split()[0].replace(',', ''))
                except (ValueError, IndexError) as e:
                    print(f"Error parsing CPU energy: {e}")
        # Debug output
        print(f"memory_energy {memory_energy} cpu_energy {cpu_energy}")
        return memory_energy, cpu_energy

    def measure_gpu(self):
        gpu_power = 0.0
        for handle in self.gpu_handles:
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                gpu_power += power_mw / 1000.0  # Convert to Watts
            except pynvml.NVMLError:
                pass
        gpu_energy = gpu_power * self.sample_interval
        return gpu_energy

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

    def run(self):
        heartbeat_thread = Thread(target=self.send_heartbeat)
        heartbeat_thread.start()
        next_time = time.time()
        next_time = next_time - (next_time % self.sample_interval)
        try:
            while not self.stop_event.is_set():
                sleep_time = next_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # We're behind schedule; adjust next_time
                    next_time = time.time()
                    next_time = next_time - (next_time % self.sample_interval)
                timestamp = next_time

                # Measure CPU and memory
                memory_energy, cpu_energy = self.run_perf_command()
                if memory_energy is not None:
                    self.last_measurements['memory_energy'] = memory_energy
                if cpu_energy is not None:
                    self.last_measurements['cpu_energy'] = cpu_energy

                # Measure GPU
                if self.gpu_available:
                    gpu_energy = self.measure_gpu()
                    if gpu_energy is not None:
                        self.last_measurements['gpu_energy'] = gpu_energy
                else:
                    self.last_measurements['gpu_energy'] = 0.0

                # Output results
                timestamp_str = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S.%f')[:-3]
                print(f"{timestamp_str}  memory: {self.last_measurements['memory_energy']} Joules  cpu: {self.last_measurements['cpu_energy']} Joules  gpu: {self.last_measurements['gpu_energy']} Joules")

                # Send data to master
                if self.client:
                    data_to_send = {
                        'node_id': self.client.node_id,
                        'timestamp': timestamp,
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

                next_time += self.sample_interval
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.stop_event.set()
        finally:
            heartbeat_thread.join()
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
