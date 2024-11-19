import socket
import threading
import json
import struct
import time
from datetime import datetime, timedelta
from typing import List
from influxdb import InfluxDBClient

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
        self.start_time = None
        self.time_interval = 0.1  # 100ms interval
        self.time_series = []  # List of timestamps
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

    def handle_control_connection(self, conn, addr):
        print(f"Control connection from {addr}")
        while not self.stop_event.is_set():
            try:
                # First, read the message length (4 bytes)
                header = self.recvall(conn, 4)
                if not header:
                    break
                msg_length = struct.unpack('!I', header)[0]
                # Then, read the message data
                msg_data = self.recvall(conn, msg_length)
                if not msg_data:
                    break
                msg = msg_data.decode()
                heartbeat = json.loads(msg)
                node_id = heartbeat.get('node_id')
                timestamp = heartbeat.get('timestamp')
                status = heartbeat.get('status')
                with self.lock:
                    self.nodes_status[node_id] = {'timestamp': timestamp, 'status': status}
                print(f"Received heartbeat from node {node_id}: status={status}, timestamp={timestamp}")
            except Exception as e:
                print(f"Error in control connection: {e}")
                break
        conn.close()
        print(f"Control connection closed from {addr}")

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
                data = json.loads(msg)
                node_id = data.get('node_id')
                measurements = data.get('measurements', [])
                self.process_measurements(node_id, measurements)
            except Exception as e:
                print(f"Error in data connection: {e}")
                break
        conn.close()
        print(f"Data connection closed from {addr}")

    def process_measurements(self, node_id, measurements):
        with self.lock:
            if self.start_time is None:
                # Set the start time when the first measurement arrives
                self.start_time = datetime.utcnow()
                print(f"Start time set to {self.start_time}")

            # Convert measurement timestamps to datetime objects
            for m in measurements:
                m['datetime'] = datetime.utcfromtimestamp(m['timestamp'])

            # Update the agent's data
            if node_id not in self.agent_data:
                self.agent_data[node_id] = []

            self.agent_data[node_id].extend(measurements)

            # Now, generate the time series up to the latest measurement
            self.update_time_series()

            # For each agent, fill missing data and write to InfluxDB
            for agent_id in self.agent_data.keys():
                self.fill_and_store_agent_data(agent_id)

    def update_time_series(self):
        # Generate time series from start_time to now with 100ms intervals
        current_time = datetime.utcnow()
        if self.time_series:
            last_time = self.time_series[-1]
        else:
            last_time = self.start_time

        while last_time <= current_time:
            last_time += timedelta(seconds=self.time_interval)
            self.time_series.append(last_time)

    def fill_and_store_agent_data(self, agent_id):
        agent_measurements = self.agent_data[agent_id]
        # Sort the agent's measurements by datetime
        agent_measurements.sort(key=lambda x: x['datetime'])

        # Create a dictionary of measurements keyed by timestamp
        measurement_dict = {m['datetime']: m for m in agent_measurements}

        # Prepare data points for InfluxDB
        data_points = []

        for ts in self.time_series:
            # If measurement exists at this timestamp, use it
            if ts in measurement_dict:
                m = measurement_dict[ts]
                cpu_core = m.get('cpu_core', 0.0)
                cpu_total = m.get('cpu_total', 0.0)
                gpu_readings = m.get('gpu_readings', [])
            else:
                # Interpolate or use last known measurement
                cpu_core, cpu_total, gpu_readings = self.interpolate_agent_data(agent_id, ts)

            # Prepare the data point
            data_point = {
                "measurement": "energy_measurements",
                "tags": {
                    "node_id": agent_id,
                },
                "time": ts.isoformat() + "Z",
                "fields": {
                    "cpu_core": float(cpu_core),
                    "cpu_total": float(cpu_total),
                    "gpu_energy": float(self.calculate_gpu_energy(gpu_readings)),
                }
            }
            data_points.append(data_point)

        # Write data points to InfluxDB
        if data_points:
            self.influx_client.write_points(data_points)
            print(f"Wrote {len(data_points)} points for agent {agent_id} to InfluxDB")

    def interpolate_agent_data(self, agent_id, timestamp):
        # Get agent measurements
        agent_measurements = self.agent_data[agent_id]
        # Find the two measurements surrounding the timestamp
        prev_m = None
        next_m = None

        for m in agent_measurements:
            if m['datetime'] <= timestamp:
                prev_m = m
            elif m['datetime'] > timestamp and next_m is None:
                next_m = m
                break

        if prev_m and next_m:
            # Linear interpolation
            delta = (next_m['datetime'] - prev_m['datetime']).total_seconds()
            if delta == 0:
                ratio = 0
            else:
                ratio = (timestamp - prev_m['datetime']).total_seconds() / delta

            cpu_core = prev_m.get('cpu_core', 0.0) + ratio * (next_m.get('cpu_core', 0.0) - prev_m.get('cpu_core', 0.0))
            cpu_total = prev_m.get('cpu_total', 0.0) + ratio * (next_m.get('cpu_total', 0.0) - prev_m.get('cpu_total', 0.0))
            gpu_readings = self.interpolate_gpu_readings(prev_m.get('gpu_readings', []), next_m.get('gpu_readings', []), ratio)
        elif prev_m:
            # Extrapolate using previous measurement
            cpu_core = prev_m.get('cpu_core', 0.0)
            cpu_total = prev_m.get('cpu_total', 0.0)
            gpu_readings = prev_m.get('gpu_readings', [])
        elif next_m:
            # Extrapolate using next measurement
            cpu_core = next_m.get('cpu_core', 0.0)
            cpu_total = next_m.get('cpu_total', 0.0)
            gpu_readings = next_m.get('gpu_readings', [])
        else:
            # No data available
            cpu_core = 0.0
            cpu_total = 0.0
            gpu_readings = []

        return cpu_core, cpu_total, gpu_readings

    def interpolate_gpu_readings(self, prev_readings, next_readings, ratio):
        # Interpolate GPU readings if possible
        if prev_readings and next_readings and len(prev_readings) == len(next_readings):
            return [p + ratio * (n - p) for p, n in zip(prev_readings, next_readings)]
        elif prev_readings:
            return prev_readings
        elif next_readings:
            return next_readings
        else:
            return []

    def calculate_gpu_energy(self, gpu_readings):
        # Calculate GPU energy from power readings
        gpu_power = sum(gpu_readings)
        gpu_energy = gpu_power * self.time_interval
        return gpu_energy

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
