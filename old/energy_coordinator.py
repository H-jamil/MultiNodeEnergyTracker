import socket
import threading
import struct
import json
import time
from datetime import datetime
import sqlite3
import os

class Coordinator:
    def __init__(self, control_port=5000, data_port=5001, db_file='energy_data.db'):
        self.control_port = control_port
        self.data_port = data_port
        self.db_file = db_file
        self.nodes_status = {}  # To track heartbeat messages
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        
        # Initialize database
        self.init_db()
        
    def init_db(self):
        self.conn = sqlite3.connect(self.db_file, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS energy_measurements (
                node_id TEXT,
                timestamp REAL,
                cpu_core REAL,
                cpu_total REAL,
                gpu_readings TEXT
            )
        ''')
        self.conn.commit()
    
    def start(self):
        # Start control server thread
        threading.Thread(target=self.control_server, daemon=True).start()
        # Start data server thread
        threading.Thread(target=self.data_server, daemon=True).start()
        
        print("Coordinator started. Listening on control port {} and data port {}".format(self.control_port, self.data_port))
        try:
            while not self.stop_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down coordinator...")
            self.stop_event.set()
            self.conn.close()
            print("Coordinator stopped.")
    
    def control_server(self):
        # Control server to handle heartbeats
        control_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        control_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        control_sock.bind(('', self.control_port))
        control_sock.listen(5)
        print("Control server listening on port {}".format(self.control_port))
        
        while not self.stop_event.is_set():
            try:
                control_sock.settimeout(1.0)
                conn, addr = control_sock.accept()
                threading.Thread(target=self.handle_control_connection, args=(conn, addr), daemon=True).start()
            except socket.timeout:
                continue
        control_sock.close()
    
    def handle_control_connection(self, conn, addr):
        print("Control connection from {}".format(addr))
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
                print("Received heartbeat from node {}: status={}, timestamp={}".format(node_id, status, timestamp))
            except Exception as e:
                print("Error in control connection: {}".format(e))
                break
        conn.close()
        print("Control connection closed from {}".format(addr))

    
    def data_server(self):
        # Data server to handle energy measurements
        data_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        data_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        data_sock.bind(('', self.data_port))
        data_sock.listen(5)
        print("Data server listening on port {}".format(self.data_port))
        
        while not self.stop_event.is_set():
            try:
                data_sock.settimeout(1.0)
                conn, addr = data_sock.accept()
                threading.Thread(target=self.handle_data_connection, args=(conn, addr), daemon=True).start()
            except socket.timeout:
                continue
        data_sock.close()
    
    def handle_data_connection(self, conn, addr):
        # Handle energy measurement data
        print("Data connection from {}".format(addr))
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
                data = json.loads(msg)
                node_id = data.get('node_id')
                measurements = data.get('measurements', [])
                self.store_measurements(node_id, measurements)
                print("Received data from node {}: {} measurements".format(node_id, len(measurements)))
            except Exception as e:
                print("Error in data connection: {}".format(e))
                break
        conn.close()
        print("Data connection closed from {}".format(addr))
    
    def recvall(self, conn, n):
        # Helper function to receive n bytes or return None if EOF is hit
        data = b''
        while len(data) < n:
            packet = conn.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data
    
    def store_measurements(self, node_id, measurements):
        with self.lock:
            for m in measurements:
                timestamp = m.get('timestamp')
                cpu_core = m.get('cpu_core')
                cpu_total = m.get('cpu_total')
                gpu_readings = m.get('gpu_readings')
                gpu_readings_str = json.dumps(gpu_readings)  # Store as JSON string
                self.cursor.execute('''
                    INSERT INTO energy_measurements (node_id, timestamp, cpu_core, cpu_total, gpu_readings)
                    VALUES (?, ?, ?, ?, ?)
                ''', (node_id, timestamp, cpu_core, cpu_total, gpu_readings_str))
            self.conn.commit()
    
    # Function to query energy consumption between t1 and t2
    def query_energy(self, t1, t2):
        with self.lock:
            self.cursor.execute('''
                SELECT node_id, timestamp, cpu_core, cpu_total, gpu_readings
                FROM energy_measurements
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
            ''', (t1, t2))
            rows = self.cursor.fetchall()
            return rows
    
    # Function to display current node statuses
    def display_node_statuses(self):
        with self.lock:
            print("\nCurrent Node Statuses:")
            for node_id, status in self.nodes_status.items():
                last_heartbeat = datetime.fromtimestamp(status['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                print("Node {}: Status={}, Last heartbeat at {}".format(node_id, status['status'], last_heartbeat))
            print()
    
    # Function to perform gap-filling (simple implementation)
    def gap_filling(self, node_id, measurements):
        # Implement gap-filling logic here if needed
        pass

if __name__ == "__main__":
    coordinator = Coordinator()
    coordinator.start()
