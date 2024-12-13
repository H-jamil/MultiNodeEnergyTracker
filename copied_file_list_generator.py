import json
import sys
from pathlib import Path

def extract_file_paths(input_file: str, output_file: str):
    file_paths = []

    # Read the input file and extract the first part of each line
    with open(input_file, 'r') as f:
        for line in f:
            # Split the line by tab and take the first part
            file_path = line.split('\t')[0].strip()
            file_paths.append(file_path)

    # Save the list of file paths to a JSON file
    with open(output_file, 'w') as f:
        json.dump(file_paths, f, indent=4)

    print(f"File paths extracted and saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_file_paths.py <input_txt_file>")
        sys.exit(1)

    input_txt_file = sys.argv[1]
    output_json_file = Path(input_txt_file).stem + "_file_paths.json"

    extract_file_paths(input_txt_file, output_json_file)