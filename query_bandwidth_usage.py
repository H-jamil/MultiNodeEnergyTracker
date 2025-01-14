import sys

def sum_bandwidth_metrics_in_gb(file_path, t1, t2):
    """
    Summation of rxbytes, txbytes, rxpackets, txpackets
    for unix_time in [t1, t2], with rxbytes & txbytes reported in GB.

    Args:
        file_path (str): Path to the .txt file containing the data.
        t1 (int): Start timestamp (inclusive).
        t2 (int): End timestamp (inclusive).

    Returns:
        dict: A dictionary with the sum of rxbytes (GB), txbytes (GB),
              rxpackets, txpackets.
    """
    total_rxbytes = 0.0
    total_txbytes = 0.0
    total_rxpackets = 0.0
    total_txpackets = 0.0

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            
            unix_time_str, interface, rxbytes_str, txbytes_str, rxpackets_str, txpackets_str = parts
            
            try:
                unix_time = int(unix_time_str)
                rxbytes = float(rxbytes_str)
                txbytes = float(txbytes_str)
                rxpackets = float(rxpackets_str)
                txpackets = float(txpackets_str)
            except ValueError:
                continue
            
            if t1 <= unix_time <= t2:
                total_rxbytes += rxbytes
                total_txbytes += txbytes
                total_rxpackets += rxpackets
                total_txpackets += txpackets

    # Convert from bytes to GB (1 GiB = 1024^3 bytes)
    bytes_to_gb = 1024.0 ** 3
    total_rx_gb = total_rxbytes / bytes_to_gb
    total_tx_gb = total_txbytes / bytes_to_gb

    return {
        'rxbytes_gb': total_rx_gb,
        'txbytes_gb': total_tx_gb,
        'rxpackets': total_rxpackets,
        'txpackets': total_txpackets
    }

def main():
    # Example usage:
    #
    #   python script.py path_to_file.txt 1735935335 1735935341
    #
    if len(sys.argv) != 4:
        print("Usage: python script.py <file_path> <start_time> <end_time>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    t1 = int(sys.argv[2])
    t2 = int(sys.argv[3])
    
    result = sum_bandwidth_metrics_in_gb(file_path, t1, t2)
    
    print(f"Sum of metrics from {t1} to {t2}:")
    print(f"  rxbytes (GB)   = {result['rxbytes_gb']:.6f}")
    print(f"  txbytes (GB)   = {result['txbytes_gb']:.6f}")
    print(f"  rxpackets      = {result['rxpackets']}")
    print(f"  txpackets      = {result['txpackets']}")

if __name__ == "__main__":
    main()

