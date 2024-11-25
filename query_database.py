import argparse
import sys
from influxdb import InfluxDBClient
from datetime import datetime
import pytz

def parse_arguments():
    parser = argparse.ArgumentParser(description="Query InfluxDB for energy measurements")
    parser.add_argument('--start', type=float, required=True, help='Start time (timestamp in seconds since epoch)')
    parser.add_argument('--end', type=float, required=True, help='End time (timestamp in seconds since epoch)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--influx_host', type=str, default='localhost', help='InfluxDB host')
    parser.add_argument('--influx_port', type=int, default=8086, help='InfluxDB port')
    parser.add_argument('--influx_db', type=str, default='energy_data', help='InfluxDB database name')
    parser.add_argument('--influx_user', type=str, default=None, help='InfluxDB username')
    parser.add_argument('--influx_pass', type=str, default=None, help='InfluxDB password')
    return parser.parse_args()

def main():
    args = parse_arguments()

    start_time = args.start
    end_time = args.end
    verbose = args.verbose

    if end_time < start_time:
        print("End time must be after start time.")
        sys.exit(1)

    duration = end_time - start_time
    # print(f"duration ={duration}")
    # Convert timestamps to datetime objects in UTC
    start_datetime = datetime.utcfromtimestamp(start_time).replace(tzinfo=pytz.UTC)
    end_datetime = datetime.utcfromtimestamp(end_time).replace(tzinfo=pytz.UTC)

    # Convert datetime objects to ISO format strings
    start_iso = start_datetime.isoformat()
    end_iso = end_datetime.isoformat()

    # Connect to InfluxDB
    influx_client = InfluxDBClient(
        host=args.influx_host,
        port=args.influx_port,
        username=args.influx_user,
        password=args.influx_pass,
        database=args.influx_db
    )

    if duration >= 0.1:
        # Fetch data within the time range
        query = f"SELECT * FROM energy WHERE time >= '{start_iso}' AND time <= '{end_iso}'"
        result = influx_client.query(query)
        points = list(result.get_points())
    else:
        # Duration less than 100ms, get the closest data point to the start time
        # Find the closest timestamp
        query_closest = f"SELECT * FROM energy WHERE time <= '{start_iso}' ORDER BY time DESC LIMIT 1"
        result_closest = influx_client.query(query_closest)
        closest_points = list(result_closest.get_points())
        if closest_points:
            closest_time = closest_points[0]['time']
            # Fetch all data points at the closest timestamp
            query = f"SELECT * FROM energy WHERE time = '{closest_time}'"
            result = influx_client.query(query)
            points = list(result.get_points())
        else:
            # No data found
            points = []

    if not points:
        print("No entries found in the specified time range.")
        sys.exit(1)

    # Remove duplicates (same timestamp and node_id)
    unique_points = {}
    for point in points:
        key = (point['time'], point.get('node_id', 'N/A'))
        if key not in unique_points:
            unique_points[key] = point

    points = list(unique_points.values())

    if verbose:
        for point in points:
            # point is a dict with keys: time, node_id, cpu_energy, memory_energy, gpu_energy
            timestamp_str = point['time']  # ISO format string
            # Parse the ISO format timestamp
            try:
                timestamp_dt = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.%fZ')
            except ValueError:
                timestamp_dt = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%SZ')
            timestamp = timestamp_dt.timestamp()  # Convert to Unix timestamp
            print(f"Time: {timestamp}, Node ID: {point.get('node_id', 'N/A')}, "
                  f"CPU Energy: {point.get('cpu_energy', 0)}, "
                  f"Memory Energy: {point.get('memory_energy', 0)}, "
                  f"GPU Energy: {point.get('gpu_energy', 0)}")
    else:
        # Compute the summary statistics
        num_entries = len(points)
        total_cpu_energy = sum(float(p.get('cpu_energy', 0)) for p in points)
        total_memory_energy = sum(float(p.get('memory_energy', 0)) for p in points)
        total_gpu_energy = sum(float(p.get('gpu_energy', 0)) for p in points)
        # Get the node_id (assuming all entries have the same node_id)
        node_id = points[0].get('node_id', 'N/A')
        # Output in one line with identifier
        print(f"{node_id}: Number of entries: {num_entries}, CPU Energy: {total_cpu_energy}, "
              f"Memory Energy: {total_memory_energy}, GPU Energy: {total_gpu_energy}")

if __name__ == "__main__":
    main()
