import socket
import json
import struct
import threading
from collections import defaultdict
from datetime import datetime
import time

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
            client_socket, addr = self.control_socket.accept()
            node_id = client_socket.recv(1024).decode()
            print(f"New node connected: {node_id} from {addr}")

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
                print(f"Error handling client data: {e}")
                break
        
        client_socket.close()

    def data_listener(self):
        self.data_socket.listen(5)
        while self.running:
            client_socket, addr = self.data_socket.accept()
            client_thread = threading.Thread(
                target=self.handle_client_data,
                args=(client_socket, addr)
            )
            self.client_threads.append(client_thread)
            client_thread.start()

    # def print_aggregated_data(self):
    #     while self.running:
    #         with self.nodes_lock:
    #             if self.nodes_data:
    #                 print("\nTimestamp            ", end="")
    #                 for node_id in sorted(self.nodes_data.keys()):
    #                     print(f"{node_id:>30}", end="")
    #                 print("\n" + "-" * (30 * (len(self.nodes_data) + 1)))

    #                 # Group measurements by timestamp
    #                 timestamp_data = defaultdict(dict)
    #                 for node_id, measurements in self.nodes_data.items():
    #                     for m in measurements:
    #                         timestamp_data[m['timestamp']][node_id] = m

    #                 # Print sorted timestamps
    #                 for timestamp in sorted(timestamp_data.keys()):
    #                     time_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    #                     print(f"{time_str:<20}", end="")
                        
    #                     for node_id in sorted(self.nodes_data.keys()):
    #                         if node_id in timestamp_data[timestamp]:
    #                             m = timestamp_data[timestamp][node_id]
    #                             readings = f"{m['cpu_core']:.1f},{m['cpu_total']:.1f}"
    #                             for gpu in m['gpu_readings']:
    #                                 readings += f",{gpu:.1f}"
    #                             print(f"{readings:>30}", end="")
    #                         else:
    #                             print(" " * 30, end="")
    #                     print()

    #                 # Clear processed data
    #                 self.nodes_data.clear()
            
    #         time.sleep(2.0)
    def print_aggregated_data(self):
        while self.running:
            with self.nodes_lock:
                if self.nodes_data:
                    # Print header
                    print("\nTimestamp                  ", end="")
                    for node_id in sorted(self.nodes_data.keys()):
                        print(f"{node_id:>30}", end="")
                    print("\n" + "-" * (30 * (len(self.nodes_data) + 1)))

                    # Group data by seconds first
                    second_groups = defaultdict(lambda: defaultdict(list))
                    
                    # Collect all data grouped by seconds
                    for node_id, measurements in self.nodes_data.items():
                        for m in measurements:
                            # Get second-level timestamp
                            second_ts = int(m['timestamp'])
                            second_groups[second_ts][node_id].append({
                                'full_ts': m['timestamp'],
                                'data': m
                            })

                    # Process each second
                    for second_ts in sorted(second_groups.keys()):
                        # Get base time string for this second
                        base_time_str = datetime.fromtimestamp(second_ts).strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Get all measurements for this second
                        second_data = second_groups[second_ts]
                        
                        # Find the maximum number of measurements in any node for this second
                        max_measurements = max(len(measurements) for measurements in second_data.values())
                        
                        # Print all measurements for this second
                        for i in range(max_measurements):
                            print(f"{base_time_str:<24}", end="")
                            
                            for node_id in sorted(self.nodes_data.keys()):
                                if node_id in second_data and i < len(second_data[node_id]):
                                    m = second_data[node_id][i]['data']
                                    readings = f"{m['cpu_core']:.1f},{m['cpu_total']:.1f}"
                                    if 'gpu_readings' in m and m['gpu_readings']:
                                        gpu_str = ','.join(f"{gpu:.1f}" for gpu in m['gpu_readings'])
                                        readings += f",{gpu_str}"
                                    print(f"{readings:>30}", end="")
                                else:
                                    print(" " * 30, end="")
                            print()
                        
                        # Add a blank line between seconds for better readability
                        print()

                    # Clear processed data
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