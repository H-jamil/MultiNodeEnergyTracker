import os
import time
import threading
import subprocess
import pynvml
import argparse
import socket
import struct
import json
from queue import Queue, Empty

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
        self.stop_event = threading.Event()
        self.measurements_a = Queue()
        self.measurements_b = Queue()
        self.result_queue = Queue() 
        self.last_measurement = {'memory_energy': 0.0, 'cpu_energy': 0.0, 'gpu_energy': 0.0}
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

        self.num_threads = 1 + int(self.gpu_available)
        self.barrier = threading.Barrier(self.num_threads)

    def run_perf_command(self):
        cmd = ["perf", "stat", "-a", "-e", "power/energy-ram/", "-e", "power/energy-pkg/", "sleep", str(self.sample_interval)]
        try:
            process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                print(f"Perf command failed with return code {process.returncode}")
                print(f"stderr: {stderr}")
                print(f"stdout: {stdout}")
                return None
        except Exception as e:
            print(f"Exception running perf command: {e}")
            return None

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
        return {'memory_energy': memory_energy, 'cpu_energy': cpu_energy}

    def measure_gpu(self):
        gpu_power = 0.0
        for handle in self.gpu_handles:
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                gpu_power += power_mw / 1000.0
            except pynvml.NVMLError:
                pass
        gpu_energy = round(gpu_power * self.sample_interval, 2)
        return {'gpu_energy': gpu_energy}

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

    def thread_function(self, name, barrier, measurements_queue, sample_interval=0.1):
        while not self.stop_event.is_set():
            barrier.wait()
            timestamp = time.time()
            rounded_timestamp = round(sample_interval * round(timestamp / sample_interval), 1)
            if name == 'cpu_memory':
                measurement = self.run_perf_command()
            elif name == 'gpu':
                measurement = self.measure_gpu()
            else:
                measurement = {0}
            measurements_queue.put((rounded_timestamp, measurement))
            next_time = timestamp + sample_interval
            sleep_time = next_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

    def fill_missing_timestamps(self, list_of_tuples, missing_timestamps):
        """
        Fills the missing timestamps in the list of tuples with measurements copied
        from the nearest timestamp.
        """
        # Sort the original list by timestamps
        list_of_tuples.sort(key=lambda x: x[0])

        # Combine original timestamps and measurements into a dictionary for easy access
        timestamp_to_measurement = {timestamp: measurement for timestamp, measurement in list_of_tuples}

        # Process each missing timestamp
        for missing in sorted(missing_timestamps):
            # Find the closest timestamp in the existing list
            closest_timestamp = min(timestamp_to_measurement.keys(), key=lambda t: abs(t - missing))
            # Copy the measurement of the closest timestamp to the missing timestamp
            timestamp_to_measurement[missing] = timestamp_to_measurement[closest_timestamp]

        # Create the new list of tuples, sorted by timestamps
        filled_list = sorted(timestamp_to_measurement.items(), key=lambda x: x[0])
        return filled_list

    # def process_results(self):
    #     """
    #     Periodically dequeue items from the result queue, handle missing timestamps,
    #     and send the data to the coordinator.
    #     """
    #     while not self.stop_event.is_set():
    #         try:
    #             batch = []
    #             while len(batch) < 20:
    #                 timestamp, measurement = self.result_queue.get(timeout=1)
    #                 batch.append((timestamp, measurement))
    #         except Empty:
    #             pass

    #         if not batch:
    #             continue
    #         batch.sort(key=lambda x: x[0])
    #         timestamps = [entry[0] for entry in batch]
    #         missed_timestamps=[]
    #         for i in range(len(timestamps) - 1):
    #             if timestamps[i + 1] * 10 - timestamps[i] * 10 == 2:
    #                 missed_timestamps.append((timestamps[i] * 10 + 1)/10)
    #         filled_batch=self.fill_missing_timestamps(batch,missed_timestamps)
    #         # Send to the coordinator
    #         for timestamp, measurement in filled_batch:
    #             data_packet = {
    #                 'node_id': self.client.node_id,
    #                 'timestamp': timestamp,
    #                 'measurement': measurement
    #             }
    #             try:
    #                 json_data = json.dumps(data_packet)
    #                 msg_length = struct.pack('!I', len(json_data))
    #                 self.client.data_socket.sendall(msg_length + json_data.encode())
    #                 print(f"Sent data: {data_packet}")
    #             except Exception as e:
    #                 print(f"Error sending data to coordinator: {e}")
    #                 break

    def process_results(self):
        """
        Periodically dequeue items from the result queue, handle missing timestamps,
        and send the data to the coordinator in one message.
        """
        while not self.stop_event.is_set():
            try:
                batch = []
                while len(batch) < 20:
                    timestamp, measurement = self.result_queue.get(timeout=1)
                    batch.append((timestamp, measurement))
            except Empty:
                pass

            if not batch:
                continue

            # Sort the batch by timestamp
            batch.sort(key=lambda x: x[0])

            # Detect and fill missing timestamps
            timestamps = [entry[0] for entry in batch]
            missed_timestamps = []
            for i in range(len(timestamps) - 1):
                if timestamps[i + 1] * 10 - timestamps[i] * 10 == 2:
                    missed_timestamps.append((timestamps[i] * 10 + 1) / 10)
            filled_batch = self.fill_missing_timestamps(batch, missed_timestamps)

            # Prepare the consolidated data packet
            consolidated_data = [
                {
                    'node_id': self.client.node_id,
                    'timestamp': timestamp,
                    'measurement': measurement
                }
                for timestamp, measurement in filled_batch
            ]

            # Send the consolidated batch to the coordinator
            try:
                json_data = json.dumps(consolidated_data)
                msg_length = struct.pack('!I', len(json_data))
                self.client.data_socket.sendall(msg_length + json_data.encode())
                print(f"Sent batch data of length: {len(consolidated_data)}")
            except Exception as e:
                print(f"Error sending batch data to coordinator: {e}")
    
    def accumulator_function(self, measurements_a, measurements_b, sample_interval):
        data_a = {}
        data_b = {}
        processed_timestamps = set()
        last_measurement_a = None
        last_measurement_b = None

        while not self.stop_event.is_set() or not measurements_a.empty() or not measurements_b.empty():
            try:
                while True:
                    timestamp_a, measurement_a = measurements_a.get_nowait()
                    data_a[timestamp_a] = measurement_a
            except Empty:
                pass
            try:
                while True:
                    timestamp_b, measurement_b = measurements_b.get_nowait()
                    data_b[timestamp_b] = measurement_b
            except Empty:
                pass
            all_timestamps = set(data_a.keys()) | set(data_b.keys())
            pending_timestamps = sorted(all_timestamps - processed_timestamps)
            for timestamp in pending_timestamps:
                measurement_a = data_a.get(timestamp)
                measurement_b = data_b.get(timestamp)
                if measurement_a is not None and measurement_b is not None:
                    result = (
                         measurement_a,
                         measurement_b
                    )
                    self.result_queue.put((timestamp, result))  # Enqueue the result
                    processed_timestamps.add(timestamp)
                    del data_a[timestamp]
                    del data_b[timestamp]
            time.sleep(0.01)

    def run(self):
        heartbeat_thread = threading.Thread(target=self.send_heartbeat)
        heartbeat_thread.start()
        result_processor_thread = threading.Thread(target=self.process_results)  # New thread
        result_processor_thread.start()
        try:
            thread_a = threading.Thread(target=self.thread_function, args=("cpu_memory", self.barrier, self.measurements_a, self.sample_interval))
            thread_a.start()
            if self.gpu_available:
                thread_b = threading.Thread(target=self.thread_function, args=("gpu", self.barrier, self.measurements_b, self.sample_interval))
                thread_b.start()
            else:
                thread_b = None
            accumulator_thread = threading.Thread(target=self.accumulator_function, args=(self.measurements_a, self.measurements_b if self.gpu_available else None, self.sample_interval))
            accumulator_thread.start()
            while not self.stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.stop_event.set()
        finally:
            heartbeat_thread.join()
            result_processor_thread.join()  # Join the result processor thread
            thread_a.join()
            if thread_b:
                thread_b.join()
            accumulator_thread.join()
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

    monitor = EnergyMonitor(sample_interval=args.sampling_time, client=client)
    monitor.run()