import sqlite3
from datetime import datetime, timezone
import json
import pandas as pd
import sys

def main():
    # Check if the correct number of arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python query.py '2024-11-14 20:50:52.2300' '2024-11-14 20:55:57.23'")
        print("Time format: 'YYYY-MM-DD HH:MM:SS.ssssss' (milliseconds precision)")
        sys.exit(1)

    # Get start and end times from command-line arguments
    start_time_str = sys.argv[1]
    end_time_str = sys.argv[2]

    # Convert to datetime objects with millisecond precision
    try:
        start_time_dt = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S.%f')
        end_time_dt = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        print("Incorrect time format. Use 'YYYY-MM-DD HH:MM:SS.ssssss'")
        sys.exit(1)

    # Convert to Unix timestamps with millisecond precision
    start_timestamp = start_time_dt.timestamp()
    end_timestamp = end_time_dt.timestamp()
    print(f"Querying data between {start_time_dt} and {end_time_dt}")
    # Connect to the database
    db_file = 'energy_data.db'
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Execute the query
    query = '''
        SELECT node_id, timestamp, cpu_core, cpu_total, gpu_readings
        FROM energy_measurements
        WHERE timestamp BETWEEN ? AND ?
        ORDER BY timestamp ASC
    '''
    cursor.execute(query, (start_timestamp, end_timestamp))
    rows = cursor.fetchall()

    # Close the database connection
    conn.close()

    if not rows:
        print("No data found for the specified time range.")
        sys.exit(0)

    # Parse the fetched data
    node_ids = []
    timestamps = []
    cpu_cores = []
    cpu_totals = []
    gpu_readings_list = []

    for row in rows:
        node_id, timestamp, cpu_core, cpu_total, gpu_readings = row
        node_ids.append(node_id)
        timestamps.append(datetime.fromtimestamp(timestamp))
        cpu_cores.append(cpu_core)
        cpu_totals.append(cpu_total)
        gpu_readings_list.append(json.loads(gpu_readings))

    # Create a DataFrame
    data = {
        'node_id': node_ids,
        'timestamp': timestamps,
        'cpu_core': cpu_cores,
        'cpu_total': cpu_totals,
        'gpu_readings': gpu_readings_list
    }

    df = pd.DataFrame(data)

    # Replace zero cpu_total values
    df = replace_zero_cpu_total(df)

    # Calculate total CPU core energy
    total_cpu_core_energy = df['cpu_core'].sum()
    print(f"Total CPU Core Energy: {total_cpu_core_energy:.6f} Joules")

    # Calculate total CPU package energy
    total_cpu_package_energy = df['cpu_total'].sum()
    print(f"Total CPU Package Energy: {total_cpu_package_energy:.6f} Joules")

    # Calculate total GPU energy (assuming constant power over each interval)
    sampling_interval = 0.1  # Adjust if your sampling interval differs

    # Calculate GPU energies
    df['gpu_energy'] = df['gpu_readings'].apply(lambda x: sum(x) * sampling_interval)

    # Calculate total GPU energy
    total_gpu_energy = df['gpu_energy'].sum()
    print(f"Total GPU Energy: {total_gpu_energy:.6f} Joules")

    # Display the updated DataFrame
    print(df)

def replace_zero_cpu_total(df):
    """
    Replace zero values in the 'cpu_total' column with the previous non-zero value.
    If no previous non-zero value exists, use the next non-zero value.
    """
    cpu_total_values = df['cpu_total'].values
    non_zero_indices = df['cpu_total'].ne(0).to_numpy().nonzero()[0]

    # If all values are zero or there are no non-zero values
    if len(non_zero_indices) == 0:
        print("All cpu_total values are zero. Cannot perform replacement.")
        return df

    # Iterate over the DataFrame
    for idx in range(len(cpu_total_values)):
        if cpu_total_values[idx] == 0:
            # Find the previous non-zero value
            prev_indices = non_zero_indices[non_zero_indices < idx]
            if len(prev_indices) > 0:
                prev_idx = prev_indices[-1]
                cpu_total_values[idx] = cpu_total_values[prev_idx]
            else:
                # No previous non-zero value, find the next non-zero value
                next_indices = non_zero_indices[non_zero_indices > idx]
                if len(next_indices) > 0:
                    next_idx = next_indices[0]
                    cpu_total_values[idx] = cpu_total_values[next_idx]
                else:
                    # No non-zero values at all (should not happen here)
                    pass

    df['cpu_total'] = cpu_total_values
    return df

if __name__ == '__main__':
    main()
