import re
from datetime import datetime, timedelta
from influxdb import InfluxDBClient
import pytz
import sys
import pandas as pd


def parse_log_file(log_file_path):
    batch_operations = []
    with open(log_file_path, 'r') as file:
        for line in file:
            # Match lines with batch operations
            batch_match = re.match(r'\[(\d+\.\d+)\] Epoch (\d+), Batch (\d+): (.*)', line)
            if batch_match:
                timestamp = float(batch_match.group(1))
                epoch = int(batch_match.group(2))
                batch = int(batch_match.group(3))
                details = batch_match.group(4)

                # Extract operation times
                operation_times = {}
                operations = ['Dataloader Wait Time', 'Data Transfer Time', 'Forward Pass Time', 'Backward Pass Time']
                for op in operations:
                    op_match = re.search(rf'{re.escape(op)}: Start (\d+\.\d+), End (\d+\.\d+), Duration ([\d\.]+)s', details)
                    if op_match:
                        start_time = float(op_match.group(1))
                        end_time = float(op_match.group(2))
                        duration = float(op_match.group(3))
                        operation_times[op] = {'start': start_time, 'end': end_time, 'duration': duration}
                batch_operations.append({
                    'epoch': epoch,
                    'batch': batch,
                    'operations': operation_times
                })
    return batch_operations

# def query_influxdb(start_time, end_time, client):
#     # Convert timestamps to RFC3339 format with nanosecond precision
#     start_dt = datetime.utcfromtimestamp(start_time).replace(tzinfo=pytz.UTC)
#     end_dt = datetime.utcfromtimestamp(end_time).replace(tzinfo=pytz.UTC)

#     duration = end_time - start_time
#     if duration < 0.1:
#         # Duration less than 100ms, get the closest data point to the start time
#         query = f"""
#         SELECT * FROM energy_measurements
#         WHERE time <= '{start_dt.isoformat()}'
#         ORDER BY time DESC
#         LIMIT 4
#         """
#     else:
#         # Duration greater than or equal to 100ms, get data between start and end times
#         query = f"""
#         SELECT * FROM energy_measurements
#         WHERE time >= '{start_dt.isoformat()}' AND time <= '{end_dt.isoformat()}'
#         """
#     result = client.query(query)
#     points = list(result.get_points())
#     return points


def find_adjacent_positive(data, index, variable):
    """
    Finds the nearest positive value in the same variable for a given row index.
    
    Parameters:
        data (list of dict): The energy data list.
        index (int): The index of the current row.
        variable (str): The variable to search for ('CPU Core' or 'CPU Total').
        
    Returns:
        float: The nearest positive value if found, otherwise 0.0.
    """
    # Check previous entries
    for i in range(index - 1, -1, -1):
        if data[i][variable] > 0:
            return data[i][variable]

    # Check subsequent entries
    for i in range(index + 1, len(data)):
        if data[i][variable] > 0:
            return data[i][variable]

    # If no positive value is found, return 0.0
    return 0.0

# def process_and_sum_energy(data):
#     """
#     Processes the energy data to replace zero values in CPU Total as described and calculates the sums.
    
#     Parameters:
#         data (list of dict): A list of dictionaries containing energy data with keys:
#                              - 'Time', 'Node ID', 'CPU Core', 'CPU Total', 'GPU Energy'
                             
#     Returns:
#         dict: A dictionary with the sums of CPU Core, CPU Total, and GPU Energy.
#     """
#     total_cpu_core = 0.0
#     total_cpu_total = 0.0
#     total_gpu_energy = 0.0

#     for entry in data:
#         cpu_core = entry['cpu_core']
#         cpu_total = entry['cpu_total']
#         gpu_energy = entry['gpu_energy']

#         # Replace CPU Total if it's 0.0
#         if cpu_total == 0.0:
#             # Find non-zero CPU Total values in the data
#             non_zero_cpu_totals = [e['cpu_total'] for e in data if e['cpu_total'] != 0.0]
#             if non_zero_cpu_totals:
#                 # Replace with the average of non-zero CPU Totals
#                 cpu_total = sum(non_zero_cpu_totals) / len(non_zero_cpu_totals)
#             else:
#                 # Replace with 5 times the corresponding CPU Core value
#                 cpu_total = cpu_core * 5

#         # Add to totals
#         total_cpu_core += cpu_core
#         total_cpu_total += cpu_total
#         total_gpu_energy += gpu_energy

#     return {
#         'Total CPU Core Energy': total_cpu_core,
#         'Total CPU Total Energy': total_cpu_total,
#         'Total GPU Energy': total_gpu_energy
#     }

def process_and_sum_energy(data):
    """
    Processes the energy data to replace zero or negative values in CPU Total and CPU Core as described,
    and calculates the sums of CPU Core, CPU Total, and GPU Energy.
    
    Parameters:
        data (list of dict): A list of dictionaries containing energy data with keys:
                             - 'Time', 'Node ID', 'CPU Core', 'CPU Total', 'GPU Energy'
                             
    Returns:
        dict: A dictionary with the sums of CPU Core, CPU Total, and GPU Energy.
    """
    total_cpu_core = 0.0
    total_cpu_total = 0.0
    total_gpu_energy = 0.0

    # Replace negative values with adjacent positive values
    for i in range(len(data)):
        # Replace negative CPU Core
        if data[i]['cpu_core'] < 0:
            data[i]['cpu_core'] = find_adjacent_positive(data, i, 'cpu_core')
        
        # Replace negative CPU Total
        if data[i]['cpu_total'] < 0:
            data[i]['cpu_total'] = find_adjacent_positive(data, i, 'cpu_total')

    for entry in data:
        cpu_core = entry['cpu_core']
        cpu_total = entry['cpu_total']
        gpu_energy = entry['gpu_energy']

        # Replace zero CPU Total as per the rules
        if cpu_total == 0.0:
            # Find non-zero CPU Total values in the data
            non_zero_cpu_totals = [e['cpu_total'] for e in data if e['cpu_total'] != 0.0]
            if non_zero_cpu_totals:
                # Replace with the average of non-zero CPU Totals
                cpu_total = sum(non_zero_cpu_totals) / len(non_zero_cpu_totals)
            else:
                # Replace with 5 times the corresponding CPU Core value
                cpu_total = cpu_core * 5

        # Add to totals
        total_cpu_core += cpu_core
        total_cpu_total += cpu_total
        total_gpu_energy += gpu_energy

    return {
        'Total CPU Core Energy': total_cpu_core,
        'Total CPU Total Energy': total_cpu_total,
        'Total GPU Energy': total_gpu_energy
    }



def query_influxdb(start_time, end_time, client):
    # client = InfluxDBClient(host=INFLUXDB_HOST, port=INFLUXDB_PORT, database=INFLUXDB_DATABASE)
    # Convert timestamps to RFC3339 format
    start_datetime = datetime.utcfromtimestamp(start_time).replace(tzinfo=pytz.UTC)
    end_datetime = datetime.utcfromtimestamp(end_time).replace(tzinfo=pytz.UTC)
    # Handle intervals less than 100ms by fetching all data points at the closest timestamp
    # time_diff = (end_datetime - start_datetime).total_seconds()
    time_diff = end_time - start_time
    if time_diff < 0.1:
        # Step 1: Find the closest timestamp
        # Duration less than 100ms, get the closest data point to the start time
        query_closest = f"""
        SELECT * FROM energy_measurements
        WHERE time <= '{start_datetime.isoformat()}'
        ORDER BY time DESC
        LIMIT 1
        """
        result_closest = client.query(query_closest)
        closest_points = list(result_closest.get_points())
        if closest_points:
            closest_time = closest_points[0]['time']
            # Step 2: Fetch all data points at the closest timestamp
            query = f"""
            SELECT * FROM energy_measurements
            WHERE time = '{closest_time}'
            """
            result = client.query(query)
            points = list(result.get_points())
        else:
            # No data found
            points = []
    else:
        # Fetch data within the time range
        # Duration greater than or equal to 100ms, get data between start and end times
        query = f"""
        SELECT * FROM energy_measurements
        WHERE time >= '{start_datetime.isoformat()}' AND time <= '{end_datetime.isoformat()}'
        """
        result = client.query(query)
        points = list(result.get_points())

    # Remove duplicates (same timestamp and node_id)
    unique_points = {}
    for point in points:
        key = (point['time'], point['node_id'])
        if key not in unique_points:
            unique_points[key] = point
        else:
            # Duplicate found; skip adding
            pass

    # Return the unique points as a list
    return list(unique_points.values())


def main():
    if len(sys.argv) != 3:
        print("Usage: python query_analysis.py logfile csvFile")
        sys.exit(1)
    log_file_path = sys.argv[1]
    csv_file_path = sys.argv[2]

    batch_ops = parse_log_file(log_file_path)
    # print(batch_ops)
    print(f"total {len(batch_ops)} batches were found")
    client = InfluxDBClient(host='localhost', port=8086, database='energy_data')
    results_csv=[]
    for batch in batch_ops:
        batch_data={}
        epoch = batch['epoch']
        batch_data['epoch']=epoch
        batch_num = batch['batch']
        batch_data['batch_num']=batch_num
        print(f"\nEpoch {epoch}, Batch {batch_num}")

        for op_name, times in batch['operations'].items():
            start_time = times['start']
            batch_data['op_name']=op_name
            end_time = times['end']
            duration = times['duration']
            op_duration=f"{op_name}_duration"
            batch_data[op_duration]=duration
            print(f"\nOperation: {op_name}")
            print(f"Start Time: {datetime.utcfromtimestamp(start_time)}")
            print(f"End Time: {datetime.utcfromtimestamp(end_time)}")
            print(f"Duration: {duration} seconds")

            data_points = query_influxdb(start_time, end_time, client)

            if data_points:
                print(f"Retrieved {len(data_points)} data points:")
                for point in data_points:
                    print(f"Time: {point['time']}, Node ID: {point['node_id']}, CPU Core: {point['cpu_core']}, CPU Total: {point['cpu_total']}, GPU Energy: {point['gpu_energy']}")

            else:
                print("No data points found for this operation.")
            cpu_gpu_energy_values =process_and_sum_energy(data_points)
            print(f"Total CPU Core Energy: {cpu_gpu_energy_values['Total CPU Core Energy']:.6f} Joules")
            print(f"Total CPU Total Energy: {cpu_gpu_energy_values['Total CPU Total Energy']:.6f} Joules")
            print(f"Total GPU Energy: {cpu_gpu_energy_values['Total GPU Energy']:.6f} Joules")
            op_cpu_core=f"{op_name}_cpu_core_energy"
            op_cpu_total=f"{op_name}_cpu_total_energy"
            op_gpu=f"{op_name}_gpu_energy"
            batch_data[op_cpu_core]=cpu_gpu_energy_values['Total CPU Core Energy']
            batch_data[op_cpu_total]=cpu_gpu_energy_values['Total CPU Total Energy']
            batch_data[op_gpu]=cpu_gpu_energy_values['Total GPU Energy']
        print(batch_data)
        results_csv.append(batch_data)
    client.close()

    column_mapping = {
    'epoch': 'E',
    'batch_num': 'B',
    'op_name': 'Op',
    'Dataloader Wait Time_duration': 'DLDuration',
    'Dataloader Wait Time_cpu_core_energy': 'DLCpuCore',
    'Dataloader Wait Time_cpu_total_energy': 'DlCpuTotal',
    'Dataloader Wait Time_gpu_energy': 'DLGpu',
    'Data Transfer Time_duration': 'DCopyDuration',
    'Data Transfer Time_cpu_core_energy': 'DCopyCpuCore',
    'Data Transfer Time_cpu_total_energy': 'DCopyCpuTotal',
    'Data Transfer Time_gpu_energy': 'DCopyGpu',
    'Forward Pass Time_duration': 'FPDuration',
    'Forward Pass Time_cpu_core_energy': 'FPCpuCore',
    'Forward Pass Time_cpu_total_energy': 'FPCpuTotal',
    'Forward Pass Time_gpu_energy': 'FPGpu',
    'Backward Pass Time_duration': 'BPDuration',
    'Backward Pass Time_cpu_core_energy': 'BPCpuCore',
    'Backward Pass Time_cpu_total_energy': 'BPCpuTotal',
    'Backward Pass Time_gpu_energy': 'BPGpu'
    }


    df = pd.DataFrame(results_csv)
    df.rename(columns=column_mapping, inplace=True)
    df.to_csv(csv_file_path, index=False)
    print(f"CSV file {csv_file_path} created successfully.")

if __name__ == '__main__':
    main()
