import sqlite3
from datetime import datetime
import pandas as pd
import sys

def main():
    # Check if the correct number of arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python query.py <start_timestamp> <end_timestamp>")
        print("Timestamp format: Unix timestamp (seconds since epoch)")
        sys.exit(1)

    # Get start and end times from command-line arguments
    start_timestamp_str = sys.argv[1]
    end_timestamp_str = sys.argv[2]

    try:
        start_timestamp = float(start_timestamp_str)
        end_timestamp = float(end_timestamp_str)
    except ValueError:
        print("Invalid timestamp format. Please provide Unix timestamps as floating point numbers.")
        sys.exit(1)

    # Convert timestamps to datetime objects for display
    start_time_dt = datetime.fromtimestamp(start_timestamp)
    end_time_dt = datetime.fromtimestamp(end_timestamp)

    print(f"Querying data between {start_time_dt} and {end_time_dt}")
    # Connect to the database
    db_file = 'energy_data_b_128_w_01_20241119_015854.db'
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Execute the query
    query = '''
        SELECT timestamp, cpu_core, cpu_total, gpu_energy
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
    gpu_energies = []

    for row in rows:
        # print(row)
        timestamp, cpu_core, cpu_total, gpu_energy = row
        # node_id, timestamp, cpu_core, cpu_total, gpu_energy = row
        # node_ids.append(node_id)
        timestamps.append(datetime.fromtimestamp(timestamp))
        cpu_cores.append(cpu_core)
        cpu_totals.append(cpu_total)
        gpu_energies.append(gpu_energy)

    # Create a DataFrame
    data = {
        # 'node_id': node_ids,
        'timestamp': timestamps,
        'cpu_core': cpu_cores,
        'cpu_total': cpu_totals,
        'gpu_energy': gpu_energies
    }

    df = pd.DataFrame(data)

    # Replace zero cpu_total values
    # df = replace_zero_cpu_total(df)

    # Calculate total CPU core energy
    total_cpu_core_energy = df['cpu_core'].sum()
    print(f"Total CPU Core Energy: {total_cpu_core_energy:.6f} Joules")

    # Calculate total CPU package energy
    total_cpu_package_energy = df['cpu_total'].sum()
    print(f"Total CPU Package Energy: {total_cpu_package_energy:.6f} Joules")

    # Calculate total GPU energy (already in Joules)
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
