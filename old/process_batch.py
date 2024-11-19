import sys
import re
import sqlite3
import pandas as pd
import csv
# import json

def main():
    if len(sys.argv) != 4:
        print("Usage: python process_batch.py output_txt_file database_file output_csv_file")
        sys.exit(1)

    output_txt_file = sys.argv[1]
    database_file = sys.argv[2]
    output_csv_file = sys.argv[3]

    # Parse the output_txt_file
    batch_data = parse_output_txt(output_txt_file)

    # Connect to the database
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    # For each batch, compute energy consumption
    results = []

    for batch in batch_data:
        batch_number = batch['batch_number']
        # DataLoad
        data_load_start = batch['data_load_start']
        data_load_end = batch['data_load_end']
        data_load_duration = data_load_end - data_load_start
        data_load_energy = compute_energy(cursor, data_load_start, data_load_end)
        # Forward Pass
        fwpass_start = batch['forward_pass_start']
        fwpass_end = batch['forward_pass_end']
        fwpass_duration = fwpass_end - fwpass_start
        fwpass_energy = compute_energy(cursor, fwpass_start, fwpass_end)
        # Backward Pass
        bwpass_start = batch['backward_pass_start']
        bwpass_end = batch['backward_pass_end']
        bwpass_duration = bwpass_end - bwpass_start
        bwpass_energy = compute_energy(cursor, bwpass_start, bwpass_end)

        results.append({
            'Batch': batch_number,
            'DataLoadStart': data_load_start,
            'DataLoadEnd': data_load_end,
            'DataLoadDURATION': data_load_duration,
            'DataLoadENERGYCPU': data_load_energy['cpu_core'],
            'DataLoadENERGYCPUTOTAL': data_load_energy['cpu_total'],
            'DataLoadENERGYGPU': data_load_energy['gpu'],
            'fwpassStart': fwpass_start,
            'fwpassEnd': fwpass_end,
            'fwpassDURATION': fwpass_duration,
            'fwpassENERGYCPU': fwpass_energy['cpu_core'],
            'fwpassENERGYCPUTOTAL': fwpass_energy['cpu_total'],
            'fwpassENERGYGPU': fwpass_energy['gpu'],
            'bwpassStart': bwpass_start,
            'bwpassEnd': bwpass_end,
            'bwpassDURATION': bwpass_duration,
            'bwpassENERGYCPU': bwpass_energy['cpu_core'],
            'bwpassENERGYCPUTOTAL': bwpass_energy['cpu_total'],
            'bwpassENERGYGPU': bwpass_energy['gpu'],
        })

    # Close the database connection
    conn.close()

    # Write results to CSV
    fieldnames = [
        'Batch', 'DataLoadStart', 'DataLoadEnd', 'DataLoadDURATION',
        'DataLoadENERGYCPU', 'DataLoadENERGYCPUTOTAL', 'DataLoadENERGYGPU',
        'fwpassStart', 'fwpassEnd', 'fwpassDURATION',
        'fwpassENERGYCPU', 'fwpassENERGYCPUTOTAL', 'fwpassENERGYGPU',
        'bwpassStart', 'bwpassEnd', 'bwpassDURATION',
        'bwpassENERGYCPU', 'bwpassENERGYCPUTOTAL', 'bwpassENERGYGPU'
    ]

    with open(output_csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

def parse_output_txt(output_txt_file):
    batch_data = []

    # Regular expressions to parse the lines
    batch_line_re = re.compile(
        r'\[(?P<timestamp>\d+\.\d+)\] Epoch (?P<epoch>\d+), Batch (?P<batch_number>\d+): '
        r'Dataloader Wait Time: Start (?P<data_load_start>\d+\.\d+), End (?P<data_load_end>\d+\.\d+), Duration (?P<data_load_duration>[\d\.]+)s; '
        r'Data Transfer Time: Start (?P<data_transfer_start>\d+\.\d+), End (?P<data_transfer_end>\d+\.\d+), Duration (?P<data_transfer_duration>[\d\.]+)s; '
        r'Forward Pass Time: Start (?P<forward_pass_start>\d+\.\d+), End (?P<forward_pass_end>\d+\.\d+), Duration (?P<forward_pass_duration>[\d\.]+)s; '
        r'Backward Pass Time: Start (?P<backward_pass_start>\d+\.\d+), End (?P<backward_pass_end>\d+\.\d+), Duration (?P<backward_pass_duration>[\d\.]+)s; '
        r'Total Iteration Time: (?P<total_iteration_time>[\d\.]+)s'
    )

    with open(output_txt_file, 'r') as f:
        for line in f:
            match = batch_line_re.match(line.strip())
            if match:
                data = match.groupdict()
                batch = {
                    'batch_number': int(data['batch_number']),
                    'data_load_start': float(data['data_load_start']),
                    'data_load_end': float(data['data_load_end']),
                    'forward_pass_start': float(data['forward_pass_start']),
                    'forward_pass_end': float(data['forward_pass_end']),
                    'backward_pass_start': float(data['backward_pass_start']),
                    'backward_pass_end': float(data['backward_pass_end']),
                }
                batch_data.append(batch)

    return batch_data

# def compute_energy(cursor, start_time, end_time):
#     # Original duration
#     original_duration = end_time - start_time

#     # Maximum extension limits
#     extended_start_time = start_time - 2 * original_duration
#     extended_end_time = end_time + 2 * original_duration

#     # Initial query within the original time range
#     query = '''
#         SELECT timestamp, cpu_core, cpu_total, gpu_energy
#         FROM energy_measurements
#         WHERE timestamp BETWEEN ? AND ?
#         ORDER BY timestamp ASC
#     '''
#     cursor.execute(query, (start_time, end_time))
#     rows = cursor.fetchall()

#     if not rows:
#         print(f"No data found for the specified time range: {start_time} to {end_time}")
#         return {'cpu_core': 0.0, 'cpu_total': 0.0, 'gpu': 0.0}

#     # Parse the fetched data into a DataFrame
#     data = {
#         'timestamp': [],
#         'cpu_core': [],
#         'cpu_total': [],
#         'gpu_energy': []
#     }

#     for row in rows:
#         timestamp, cpu_core, cpu_total, gpu_energy = row
#         data['timestamp'].append(timestamp)
#         data['cpu_core'].append(cpu_core)
#         data['cpu_total'].append(cpu_total)
#         data['gpu_energy'].append(gpu_energy)

#     df = pd.DataFrame(data)

#     # Count non-zero data points within the original time range
#     n1 = (df['cpu_core'] != 0).sum()
#     n2 = (df['cpu_total'] != 0).sum()
#     n3 = (df['gpu_energy'] != 0).sum()
#     N = max(n1, n2, n3)

#     # Function to extend data for a variable
#     def extend_variable(variable_name, current_df):
#         non_zero_count = (current_df[variable_name] != 0).sum()
#         required = N - non_zero_count
#         if required <= 0:
#             return current_df  # No extension needed

#         # Query additional data points outside the original time range
#         additional_rows = []

#         # First, query before the start_time
#         if required > 0:
#             cursor.execute('''
#                 SELECT timestamp, cpu_core, cpu_total, gpu_energy
#                 FROM energy_measurements
#                 WHERE timestamp BETWEEN ? AND ?
#                 ORDER BY timestamp DESC
#             ''', (extended_start_time, start_time))
#             rows_before = cursor.fetchall()

#             for row in rows_before:
#                 timestamp, cpu_core, cpu_total, gpu_energy = row
#                 additional_rows.append({
#                     'timestamp': timestamp,
#                     'cpu_core': cpu_core,
#                     'cpu_total': cpu_total,
#                     'gpu_energy': gpu_energy
#                 })
#                 if len(additional_rows) >= required:
#                     break  # Stop if we've collected enough data

#         # Then, query after the end_time if still needed
#         if len(additional_rows) < required:
#             cursor.execute('''
#                 SELECT timestamp, cpu_core, cpu_total, gpu_energy
#                 FROM energy_measurements
#                 WHERE timestamp BETWEEN ? AND ?
#                 ORDER BY timestamp ASC
#             ''', (end_time, extended_end_time))
#             rows_after = cursor.fetchall()

#             for row in rows_after:
#                 timestamp, cpu_core, cpu_total, gpu_energy = row
#                 additional_rows.append({
#                     'timestamp': timestamp,
#                     'cpu_core': cpu_core,
#                     'cpu_total': cpu_total,
#                     'gpu_energy': gpu_energy
#                 })
#                 if len(additional_rows) >= required:
#                     break  # Stop if we've collected enough data

#         # Create a DataFrame from the additional rows
#         columns = ['timestamp', 'cpu_core', 'cpu_total', 'gpu_energy']
#         additional_df = pd.DataFrame(additional_rows, columns=columns)

#         # Ensure DataFrame has the expected columns even if empty
#         if additional_df.empty:
#             print(f"Warning: No additional data found for variable '{variable_name}' in extended time range.")
#             return current_df

#         # Filter for non-zero values of the variable
#         additional_non_zero = additional_df[additional_df[variable_name] != 0]

#         if additional_non_zero.empty:
#             print(f"Warning: No additional non-zero data found for variable '{variable_name}' in extended time range.")
#             return current_df

#         # Select up to 'required' number of data points
#         additional_non_zero = additional_non_zero.head(required)

#         # Append to the current DataFrame
#         extended_df = pd.concat([current_df, additional_non_zero], ignore_index=True)

#         return extended_df

#     # Extend variables with fewer non-zero data points
#     variables = ['cpu_core', 'cpu_total', 'gpu_energy']
#     for var in variables:
#         df = extend_variable(var, df)

#     # After extension, ensure we have N non-zero data points for each variable
#     for var in variables:
#         non_zero_count = (df[var] != 0).sum()
#         if non_zero_count < N:
#             print(f"Warning: Could not find enough non-zero data points for '{var}'. Found {non_zero_count}, required {N}.")

#     # Replace zero cpu_total values
#     df = replace_zero_cpu_total(df)

#     # Calculate total energies
#     total_cpu_core_energy = df['cpu_core'].sum()
#     total_cpu_package_energy = df['cpu_total'].sum()
#     total_gpu_energy = df['gpu_energy'].sum()

#     return {
#         'cpu_core': total_cpu_core_energy,
#         'cpu_total': total_cpu_package_energy,
#         'gpu': total_gpu_energy
#     }

def compute_energy(cursor, start_time, end_time):
    # Query the database for energy measurements between start_time and end_time
    query = '''
        SELECT timestamp, cpu_core, cpu_total, gpu_energy
        FROM energy_measurements
        WHERE timestamp BETWEEN ? AND ?
        ORDER BY timestamp ASC
    '''
    cursor.execute(query, (start_time, end_time))
    rows = cursor.fetchall()

    if not rows:
        print(f"No data found for the specified time range: {start_time} to {end_time}")
        return {'cpu_core': 0.0, 'cpu_total': 0.0, 'gpu': 0.0}

    # Parse the fetched data
    timestamps = []
    cpu_cores = []
    cpu_totals = []
    gpu_readings_list = []

    for row in rows:
        timestamp, cpu_core, cpu_total, gpu_readings = row
        timestamps.append(timestamp)
        cpu_cores.append(cpu_core)
        cpu_totals.append(cpu_total)
        gpu_readings_list.append(gpu_readings)

    # Create a DataFrame
    data = {
        'timestamp': timestamps,
        'cpu_core': cpu_cores,
        'cpu_total': cpu_totals,
        'gpu_readings': gpu_readings_list
    }

    df = pd.DataFrame(data)

    # Replace zero cpu_total values
    df = replace_zero_cpu_total(df)

    # Now, calculate delta_time between samples
    # We need to include start_time and end_time in the timestamps

    # Insert start_time and end_time into the dataframe
    # df = df.sort_values('timestamp')
    # if df['timestamp'].iloc[0] > start_time:
    #     # Insert a row at the beginning
    #     df = pd.concat([pd.DataFrame({'timestamp': [start_time], 'cpu_core': [0], 'cpu_total': [0], 'gpu_readings': [[]]}), df], ignore_index=True)
    # if df['timestamp'].iloc[-1] < end_time:
    #     # Append a row at the end
    #     df = pd.concat([df, pd.DataFrame({'timestamp': [end_time], 'cpu_core': [0], 'cpu_total': [0], 'gpu_readings': [[]]})], ignore_index=True)

    # df = df.sort_values('timestamp').reset_index(drop=True)

    # # Now, calculate delta_time
    # df['delta_time'] = df['timestamp'].shift(-1) - df['timestamp']
    # df['delta_time'].iloc[-1] = end_time - df['timestamp'].iloc[-1]

    # # For each interval, the energy is the measurement at time t multiplied by delta_time until next measurement
    # # For the last measurement, delta_time is end_time - timestamp

    # # For GPU energy
    # df['gpu_power'] = df['gpu_readings'].apply(lambda x: sum(x))

    # Calculate energies
    total_cpu_core_energy = df['cpu_core'].sum()
    total_cpu_package_energy = df['cpu_total'].sum()
    # total_gpu_energy = (df['gpu_power'] * df['delta_time']).sum()
    total_gpu_energy = df['gpu_readings'].sum()

    return {'cpu_core': total_cpu_core_energy, 'cpu_total': total_cpu_package_energy, 'gpu': total_gpu_energy}


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
