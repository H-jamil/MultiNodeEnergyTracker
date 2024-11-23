import threading
import time
from queue import Queue, Empty
import random
import pynvml

class EnergyMonitor:
    def __init__(self, sample_interval=0.1):
        self.sample_interval = sample_interval
        self.stop_event = threading.Event()

        # Queues for communication
        self.timestamp_queue = Queue()
        self.cpu_memory_queue = Queue()
        self.gpu_queue = Queue()

        # Last known measurements
        self.last_cpu_memory_measurement = None
        self.last_gpu_measurement = None

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

    def cpu_memory_measurement_thread(self):
        while not self.stop_event.is_set():
            # Simulate measurement delay
            time.sleep(self.sample_interval / 2)
            timestamp = round(time.time(), 1)
            # Simulate CPU and memory energy measurements
            cpu_energy = random.uniform(5.0, 10.0)
            memory_energy = random.uniform(1.0, 2.0)
            measurement = {'timestamp': timestamp, 'cpu_energy': cpu_energy, 'memory_energy': memory_energy}
            self.cpu_memory_queue.put(measurement)

    def gpu_measurement_thread(self):
        while not self.stop_event.is_set():
            # Simulate measurement delay
            time.sleep(self.sample_interval / 2)
            timestamp = round(time.time(), 1)
            # Simulate GPU energy measurement
            gpu_energy = random.uniform(1.0, 3.0)
            measurement = {'timestamp': timestamp, 'gpu_energy': gpu_energy}
            self.gpu_queue.put(measurement)

    def timestamp_generator_thread(self):
        next_timestamp = round(time.time(), 1)
        while not self.stop_event.is_set():
            timestamp = next_timestamp
            self.timestamp_queue.put(timestamp)
            time.sleep(self.sample_interval)
            next_timestamp = round(next_timestamp + self.sample_interval, 1)
            # Adjust for floating point arithmetic issues
            next_timestamp = round(next_timestamp, 1)

    def accumulator_thread(self):
        cpu_memory_measurements = {}
        gpu_measurements = {}

        while not self.stop_event.is_set():
            try:
                timestamp = self.timestamp_queue.get(timeout=1)
            except Empty:
                continue

            # Fetch CPU/Memory measurement for the timestamp
            cpu_memory_measurement = None
            while True:
                try:
                    measurement = self.cpu_memory_queue.get_nowait()
                    cpu_memory_measurements[measurement['timestamp']] = measurement
                except Empty:
                    break
            cpu_memory_measurement = cpu_memory_measurements.pop(timestamp, None)

            # Fetch GPU measurement for the timestamp
            gpu_measurement = None
            while True:
                try:
                    measurement = self.gpu_queue.get_nowait()
                    gpu_measurements[measurement['timestamp']] = measurement
                except Empty:
                    break
            gpu_measurement = gpu_measurements.pop(timestamp, None)

            # Use last known measurements if current measurement is missing
            if cpu_memory_measurement is None:
                cpu_memory_measurement = self.last_cpu_memory_measurement
            else:
                self.last_cpu_memory_measurement = cpu_memory_measurement

            if gpu_measurement is None:
                gpu_measurement = self.last_gpu_measurement
            else:
                self.last_gpu_measurement = gpu_measurement

            # Combine measurements
            combined_measurement = {'timestamp': timestamp}
            if cpu_memory_measurement:
                combined_measurement.update(cpu_memory_measurement)
            if gpu_measurement:
                combined_measurement.update(gpu_measurement)

            # Print the combined measurement
            print(f"Time: {timestamp}, Measurements: {combined_measurement}")

            # Clean up old entries to prevent memory growth
            # Optionally, implement a time-based or size-based cache cleanup

    def run(self):
        # Start threads
        threads = []

        cpu_thread = threading.Thread(target=self.cpu_memory_measurement_thread)
        threads.append(cpu_thread)

        if self.gpu_available:
            gpu_thread = threading.Thread(target=self.gpu_measurement_thread)
            threads.append(gpu_thread)

        timestamp_thread = threading.Thread(target=self.timestamp_generator_thread)
        threads.append(timestamp_thread)

        accumulator_thread = threading.Thread(target=self.accumulator_thread)
        threads.append(accumulator_thread)

        for thread in threads:
            thread.start()

        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping...")
            self.stop_event.set()
            for thread in threads:
                thread.join()

if __name__ == "__main__":
    monitor = EnergyMonitor(sample_interval=0.1)
    monitor.run()
