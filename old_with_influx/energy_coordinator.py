import socket
import threading
import json
import struct
import time
from datetime import datetime, timedelta
from typing import List
from influxdb import InfluxDBClient
import math

class EnergyCoordinator:
    def __init__(self, control_port=5000, data_port=5001, db_identifier='default'):
        self.control_port = control_port
        self.data_port = data_port
        self.db_identifier = db_identifier
        self.control_server = None
        self.data_server = None
        self.stop_event = threading.Event()
        self.nodes_status = {}
        self.lock = threading.Lock()

        # InfluxDB setup
        self.influx_client = None
        self.setup_influxdb()

        # Time series management
        self.time_interval = 0.1  # 100ms interval
        self.agent_data = {}  # Dict to store data per agent

    def setup_influxdb(self):
        # Connect to InfluxDB
        self.influx_client = InfluxDBClient(host='localhost', port=8086, database='energy_data')
        print("Connected to InfluxDB database: energy_data")

    def start(self):
        self.control_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.control_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.control_server.bind(('', self.control_port))
        self.control_server.listen(5)
        print(f"Control server listening on port {self.control_port}")

        self.data_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.data_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.data_server.bind(('', self.data_port))
        self.data_server.listen(5)
        print(f"Data server listening on port {self.data_port}")

        threading.Thread(target=self.accept_control_connections, daemon=True).start()
        threading.Thread(target=self.accept_data_connections, daemon=True).start()

    def accept_control_connections(self):
        while not self.stop_event.is_set():
            try:
                conn, addr = self.control_server.accept()
                threading.Thread(target=self.handle_control_connection, args=(conn, addr), daemon=True).start()
            except Exception as e:
                print(f"Error accepting control connection: {e}")

    def accept_data_connections(self):
        while not self.stop_event.is_set():
            try:
                conn, addr = self.data_server.accept()
                threading.Thread(target=self.handle_data_connection, args=(conn, addr), daemon=True).start()
            except Exception as e:
                print(f"Error accepting data connection: {e}")

    def recvall(self, conn, n):
        # Helper function to receive n bytes or return None if EOF is hit
        data = b''
        while len(data) < n:
            packet = conn.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    # def handle_data_connection(self, conn, addr):
    #     print(f"Data connection from {addr}")
    #     while not self.stop_event.is_set():
    #         try:
    #             # Read message length
    #             header = self.recvall(conn, 4)
    #             if not header:
    #                 break
    #             msg_length = struct.unpack('!I', header)[0]
    #             # Read message data
    #             msg_data = self.recvall(conn, msg_length)
    #             if not msg_data:
    #                 break
    #             msg = msg_data.decode()
    #             data = json.loads(msg)
    #             node_id = data.get('node_id')
    #             timestamp = data.get('timestamp')
    #             measurement = data.get('measurement')

    #             # Extract CPU/Memory and GPU energies
    #             memory_cpu, gpu = measurement
    #             memory_energy = memory_cpu.get('memory_energy')
    #             cpu_energy = memory_cpu.get('cpu_energy')
    #             gpu_energy = gpu.get('gpu_energy')

    #             # Process the measurement
    #             self.process_measurement(node_id, timestamp, memory_energy, cpu_energy, gpu_energy)
    #         except Exception as e:
    #             print(f"Error in data connection: {e}")
    #             break
    #     conn.close()
    #     print(f"Data connection closed from {addr}")

    # def handle_data_connection(self, conn, addr):
    #     print(f"Data connection from {addr}")
    #     while not self.stop_event.is_set():
    #         try:
    #             # Read message length
    #             header = self.recvall(conn, 4)
    #             if not header:
    #                 break
    #             msg_length = struct.unpack('!I', header)[0]
    #             # Read message data
    #             msg_data = self.recvall(conn, msg_length)
    #             if not msg_data:
    #                 break
    #             msg = msg_data.decode()
    #             data = json.loads(msg)
    #             node_id = data.get('node_id')
    #             timestamp = data.get('timestamp')
    #             memory_energy = data.get('memory_energy')
    #             cpu_energy = data.get('cpu_energy')
    #             gpu_energy = data.get('gpu_energy')

    #             # Process the measurement
    #             self.process_measurement(node_id, timestamp, memory_energy, cpu_energy, gpu_energy)
    #         except Exception as e:
    #             print(f"Error in data connection: {e}")
    #             break
    #     conn.close()
    #     print(f"Data connection closed from {addr}")

    def handle_data_connection(self, conn, addr):
        print(f"Data connection from {addr}")
        while not self.stop_event.is_set():
            try:
                # Read message length
                header = self.recvall(conn, 4)
                if not header:
                    break
                msg_length = struct.unpack('!I', header)[0]
                # Read message data
                msg_data = self.recvall(conn, msg_length)
                if not msg_data:
                    break
                msg = msg_data.decode()
                batch_data = json.loads(msg)  # Parse the batch data (list of measurements)
                print(f"Received {len(batch_data)} points from {addr}")
                # Process each measurement in the batch
                for data in batch_data:
                    node_id = data.get('node_id')
                    timestamp = data.get('timestamp')
                    measurement = data.get('measurement')

                    # Extract CPU/Memory and GPU energies
                    memory_cpu, gpu = measurement
                    memory_energy = memory_cpu.get('memory_energy')
                    cpu_energy = memory_cpu.get('cpu_energy')
                    gpu_energy = gpu.get('gpu_energy')

                    # Process the measurement
                    self.process_measurement(node_id, timestamp, memory_energy, cpu_energy, gpu_energy)
            except Exception as e:
                print(f"Error in data connection: {e}")
                break
        conn.close()
        print(f"Data connection closed from {addr}")


    def process_measurement(self, node_id, timestamp, memory_energy, cpu_energy, gpu_energy):
        with self.lock:
            # Align timestamp to the sampling interval
            # aligned_timestamp = self.align_timestamp(timestamp)
            measurement = {
                'node_id': node_id,
                'timestamp': timestamp,
                'memory_energy': memory_energy,
                'cpu_energy': cpu_energy,
                'gpu_energy': gpu_energy
            }

            # Initialize data list for the node if not present
            if node_id not in self.agent_data:
                self.agent_data[node_id] = []

            # Append the measurement
            self.agent_data[node_id].append(measurement)

            # Store the data point in InfluxDB
            self.store_data_point(measurement)

    def align_timestamp(self, timestamp):
        # Align timestamp to the lower multiple of the sampling interval
        sample_interval = self.time_interval
        aligned_timestamp = math.floor(timestamp / sample_interval) * sample_interval
        return aligned_timestamp

    def store_data_point(self, measurement):
        ts = datetime.utcfromtimestamp(measurement['timestamp'])
        data_point = {
            "measurement": "energy_measurements",
            "tags": {
                "node_id": measurement['node_id'],
            },
            "time": ts.isoformat() + "Z",
            "fields": {
                "memory_energy": float(measurement['memory_energy']),
                "cpu_energy": float(measurement['cpu_energy']),
                "gpu_energy": float(measurement['gpu_energy']) if measurement['gpu_energy'] is not None else 0.0,
            }
        }
        # Write data point to InfluxDB
        self.influx_client.write_points([data_point])
        # print(f"Wrote data point for agent {measurement['node_id']} at time {ts}")

    def stop(self):
        self.stop_event.set()
        self.control_server.close()
        self.data_server.close()
        self.influx_client.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Energy Coordinator')
    parser.add_argument('--control_port', type=int, default=5000, help='Control port')
    parser.add_argument('--data_port', type=int, default=5001, help='Data port')
    parser.add_argument('--identifier', type=str, default='default', help='Database identifier')
    args = parser.parse_args()

    coordinator = EnergyCoordinator(control_port=args.control_port, data_port=args.data_port, db_identifier=args.identifier)
    coordinator.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down coordinator...")
        coordinator.stop()
