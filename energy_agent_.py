import os
import time
import threading
import subprocess
import pynvml
import argparse
from queue import Queue, Empty
from influxdb import InfluxDBClient

class EnergyMonitor:
    def __init__(self, sample_interval=0.1, node_id='compute1', influx_host='localhost', influx_port=8086, influx_db='energy_data', influx_user=None, influx_pass=None):
        self.sample_interval = sample_interval
        self.node_id = node_id
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

        # InfluxDB client
        self.influx_client = InfluxDBClient(
            host=influx_host,
            port=influx_port,
            username=influx_user,
            password=influx_pass,
            database=influx_db
        )

    def run_perf_command(self):
        # Uses perf to capture CPU and Memory energy
        cmd = ["perf", "stat", "-a", "-e", "power/energy-ram/", "-e", "power/energy-pkg/", 
               "sleep", str(self.sample_interval)]
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
        # Capture GPU energy usage
        gpu_power = 0.0
        for handle in self.gpu_handles:
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                gpu_power += power_mw / 1000.0
            except pynvml.NVMLError:
                pass
        # Convert power (mW) -> (W) and multiply by time to get energy (J)
        gpu_energy = round(gpu_power * self.sample_interval, 4)
        return {'gpu_energy': gpu_energy}

    def thread_function(self, name, barrier, measurements_queue, sample_interval=0.1):
        """
        Periodically measure either CPU/Memory or GPU energies.
        """
        while not self.stop_event.is_set():
            # Synchronize all threads at the barrier
            barrier.wait()

            timestamp = time.time()
            # Round timestamp to the nearest sample interval
            # e.g. for sample_interval=0.1, 0.2, 0.5, or 1.0
            rounded_timestamp = round(round(timestamp / sample_interval) * sample_interval, 6)

            if name == 'cpu_memory':
                measurement = self.run_perf_command()
            elif name == 'gpu':
                measurement = self.measure_gpu()
            else:
                measurement = {}

            if measurement:
                measurements_queue.put((rounded_timestamp, measurement))

            # Sleep until the next time window
            next_time = timestamp + sample_interval
            sleep_time = next_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

    def fill_missing_timestamps(self, list_of_tuples, missing_timestamps):
        """
        Fills the missing timestamps in the list of tuples with measurements 
        from the nearest existing timestamp.
        """
        # Sort by timestamps
        list_of_tuples.sort(key=lambda x: x[0])

        # Turn (timestamp, measurement) into a dict
        timestamp_to_measurement = {timestamp: measurement for timestamp, measurement in list_of_tuples}

        # Fill each missing timestamp with the nearest measurement
        for missing in sorted(missing_timestamps):
            # Find the closest existing timestamp in the dictionary
            closest_timestamp = min(timestamp_to_measurement.keys(), key=lambda t: abs(t - missing))
            timestamp_to_measurement[missing] = timestamp_to_measurement[closest_timestamp]

        # Return a new list of tuples (sorted by timestamp)
        filled_list = sorted(timestamp_to_measurement.items(), key=lambda x: x[0])
        return filled_list

    def process_results(self):
        """
        Periodically dequeue items from the result queue, handle missing timestamps,
        and write the data to InfluxDB in batches.
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
                # Check how many intervals we jumped from timestamps[i] to [i+1]
                sample_steps = int(round((timestamps[i+1] - timestamps[i]) / self.sample_interval))
                if sample_steps > 1:
                    # We have missing intervals
                    for step in range(1, sample_steps):
                        missed_timestamp = timestamps[i] + step * self.sample_interval
                        missed_timestamps.append(round(missed_timestamp, 6))

            filled_batch = self.fill_missing_timestamps(batch, missed_timestamps)

            # Prepare data points for InfluxDB
            json_body = []
            for timestamp, measurement in filled_batch:
                # measurement is (measurement_a, measurement_b)
                measurement_a, measurement_b = measurement
                fields = {}
                if measurement_a:
                    fields.update(measurement_a)
                if measurement_b:
                    fields.update(measurement_b)

                json_body.append({
                    "measurement": "energy",
                    "tags": {
                        "node_id": self.node_id
                    },
                    "time": int(timestamp * 1e9),  # Convert to nanoseconds
                    "fields": fields
                })

            # Write data points to InfluxDB
            try:
                self.influx_client.write_points(json_body)
                print(f"Wrote batch data of length: {len(filled_batch)} to InfluxDB")
            except Exception as e:
                print(f"Error writing batch data to InfluxDB: {e}")

    def accumulator_function(self, measurements_a, measurements_b, sample_interval):
        """
        Accumulate CPU/Mem data (measurements_a) & GPU data (measurements_b) 
        at the same timestamps and push into result_queue.
        """
        data_a = {}
        data_b = {}
        processed_timestamps = set()

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

            # Merge timestamps from both data_a and data_b
            all_timestamps = set(data_a.keys()) | set(data_b.keys())
            pending_timestamps = sorted(all_timestamps - processed_timestamps)

            for timestamp in pending_timestamps:
                measurement_a = data_a.get(timestamp)
                measurement_b = data_b.get(timestamp)
                # We only forward if BOTH are present
                if measurement_a is not None and measurement_b is not None:
                    result = (measurement_a, measurement_b)
                    self.result_queue.put((timestamp, result))
                    processed_timestamps.add(timestamp)
                    # Remove used data
                    if timestamp in data_a:
                        del data_a[timestamp]
                    if timestamp in data_b:
                        del data_b[timestamp]

            time.sleep(0.01)

    def run(self):
        result_processor_thread = threading.Thread(target=self.process_results)
        result_processor_thread.start()

        try:
            thread_a = threading.Thread(
                target=self.thread_function, 
                args=("cpu_memory", self.barrier, self.measurements_a, self.sample_interval)
            )
            thread_a.start()

            if self.gpu_available:
                thread_b = threading.Thread(
                    target=self.thread_function, 
                    args=("gpu", self.barrier, self.measurements_b, self.sample_interval)
                )
                thread_b.start()
            else:
                thread_b = None

            accumulator_thread = threading.Thread(
                target=self.accumulator_function, 
                args=(self.measurements_a, self.measurements_b if self.gpu_available else None, self.sample_interval)
            )
            accumulator_thread.start()

            while not self.stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.stop_event.set()
        finally:
            result_processor_thread.join()
            thread_a.join()
            if thread_b:
                thread_b.join()
            accumulator_thread.join()

            if self.gpu_available:
                pynvml.nvmlShutdown()

if __name__ == "__main__":
    if os.geteuid() != 0:
        print("This program needs root privileges to read energy values.")
        print("Please run with sudo.")
        exit(1)

    parser = argparse.ArgumentParser(description="Energy Monitoring Tool")
    parser.add_argument('--sampling_time', type=float, default=0.1, help='Sampling time in seconds (e.g. 0.1, 0.2, 0.5, 1.0)')
    parser.add_argument('--node_id', type=str, default='compute1', help='Node ID')
    parser.add_argument('--influx_host', type=str, default='localhost', help='InfluxDB host')
    parser.add_argument('--influx_port', type=int, default=8086, help='InfluxDB port')
    parser.add_argument('--influx_db', type=str, default='energy_data', help='InfluxDB database name')
    parser.add_argument('--influx_user', type=str, default=None, help='InfluxDB username')
    parser.add_argument('--influx_pass', type=str, default=None, help='InfluxDB password')
    args = parser.parse_args()

    monitor = EnergyMonitor(
        sample_interval=args.sampling_time,
        node_id=args.node_id,
        influx_host=args.influx_host,
        influx_port=args.influx_port,
        influx_db=args.influx_db,
        influx_user=args.influx_user,
        influx_pass=args.influx_pass
    )
    monitor.run()
