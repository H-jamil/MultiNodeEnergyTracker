import socket
import json
import struct
import threading
from collections import defaultdict
from datetime import datetime
import time
import statistics

class EnergyMaster:
    def __init__(self, control_port: int, data_port: int):
        self.control_port = control_port
        self.data_port = data_port
        self.nodes_data = defaultdict(list)
        self.nodes_lock = threading.Lock()
        self.running = True
        
        # Setup sockets
        self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.control_socket.bind(('0.0.0.0', control_port))
        self.data_socket.bind(('0.0.0.0', data_port))
        
        # Start listener threads
        self.client_threads = []
        threading.Thread(target=self.control_listener).start()
        threading.Thread(target=self.data_listener).start()
        threading.Thread(target=self.print_aggregated_data).start()

    def control_listener(self):
        self.control_socket.listen(5)
        while self.running:
            try:
                client_socket, addr = self.control_socket.accept()
                node_id = client_socket.recv(1024).decode()
                print(f"New node connected: {node_id} from {addr}")
            except Exception as e:
                if self.running:  # Only print error if not shutting down
                    print(f"Control listener error: {e}")

    def handle_client_data(self, client_socket, addr):
        while self.running:
            try:
                # Receive message length
                msg_len = client_socket.recv(4)
                if not msg_len:
                    break
                msg_len = struct.unpack('!I', msg_len)[0]
                
                # Receive data
                data = b''
                while len(data) < msg_len:
                    chunk = client_socket.recv(min(msg_len - len(data), 4096))
                    if not chunk:
                        break
                    data += chunk
                
                if data:
                    json_data = json.loads(data.decode())
                    with self.nodes_lock:
                        self.nodes_data[json_data['node_id']].extend(json_data['measurements'])
            except Exception as e:
                if self.running:  # Only print error if not shutting down
                    print(f"Error handling client data: {e}")
                break
        
        client_socket.close()

    def data_listener(self):
        self.data_socket.listen(5)
        while self.running:
            try:
                client_socket, addr = self.data_socket.accept()
                client_thread = threading.Thread(
                    target=self.handle_client_data,
                    args=(client_socket, addr)
                )
                self.client_threads.append(client_thread)
                client_thread.start()
            except Exception as e:
                if self.running:  # Only print error if not shutting down
                    print(f"Data listener error: {e}")

    def print_aggregated_data(self):
        WINDOW_SIZE_MS = 100  # 100ms window size for aggregation
        
        while self.running:
            with self.nodes_lock:
                if self.nodes_data:
                    # Print header
                    print("\nTimestamp                  ", end="")
                    for node_id in sorted(self.nodes_data.keys()):
                        print(f"{node_id:>30}", end="")
                    print("\n" + "-" * (30 * (len(self.nodes_data) + 1)))

                    # Create time-based buckets for each 100ms window
                    time_windows = defaultdict(lambda: defaultdict(list))
                    
                    # Collect and organize data into time windows
                    for node_id, measurements in self.nodes_data.items():
                        for m in measurements:
                            # Convert timestamp to milliseconds and round to nearest window
                            ts_ms = int(float(m['timestamp']) * 1000)
                            window_ts = (ts_ms // WINDOW_SIZE_MS) * WINDOW_SIZE_MS
                            time_windows[window_ts][node_id].append(m)

                    # Process each time window
                    for window_ts in sorted(time_windows.keys()):
                        base_time = datetime.fromtimestamp(window_ts / 1000)
                        time_str = base_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                        print(f"{time_str:<24}", end="")
                        
                        # Process each node's data within the window
                        for node_id in sorted(self.nodes_data.keys()):
                            if node_id in time_windows[window_ts]:
                                measurements = time_windows[window_ts][node_id]
                                
                                # Aggregate measurements within the window
                                avg_cpu_core = statistics.mean(m['cpu_core'] for m in measurements)
                                avg_cpu_total = statistics.mean(m['cpu_total'] for m in measurements)
                                
                                readings = f"{avg_cpu_core:.1f},{avg_cpu_total:.1f}"
                                
                                # Handle GPU readings if present
                                if any('gpu_readings' in m and m['gpu_readings'] for m in measurements):
                                    gpu_readings = [m['gpu_readings'] for m in measurements 
                                                  if 'gpu_readings' in m]
                                    if gpu_readings:
                                        avg_gpu = []
                                        for gpu_idx in range(len(gpu_readings[0])):
                                            gpu_values = [reading[gpu_idx] for reading in gpu_readings]
                                            avg_gpu.append(statistics.mean(gpu_values))
                                        gpu_str = ','.join(f"{gpu:.1f}" for gpu in avg_gpu)
                                        readings += f",{gpu_str}"
                                
                                print(f"{readings:>30}", end="")
                            else:
                                print(" " * 30, end="")
                        print()
                    
                    print()
                    self.nodes_data.clear()
            
            time.sleep(2.0)

    def shutdown(self):
        self.running = False
        self.control_socket.close()
        self.data_socket.close()
        for thread in self.client_threads:
            thread.join()

if __name__ == "__main__":
    master = EnergyMaster(control_port=5000, data_port=5001)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down master...")
        master.shutdown()