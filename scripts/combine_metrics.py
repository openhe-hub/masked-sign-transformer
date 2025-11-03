import os
import json
import csv
import glob

def compile_metrics_to_csv():
    """
    Finds all .json files in the output/metrics directory, extracts their
    'overall_metrics', and compiles them into a single CSV file.
    """
    metrics_dir = os.path.join('output', 'metrics')
    output_csv_path = os.path.join('output', 'metrics_summary.csv')
    
    # Use glob to find all json files, ignoring .gitignore rules
    json_files = glob.glob(os.path.join(metrics_dir, '*.json'))
    
    if not json_files:
        print(f"No JSON files found in {metrics_dir}. Nothing to do.")
        return

    print(f"Found {len(json_files)} JSON files to process.")

    all_metrics_data = []
    header = set()

    # First pass: read all data and determine the full set of headers
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'overall_metrics' in data and isinstance(data['overall_metrics'], dict):
                metrics = data['overall_metrics']
                # Add a column for the experiment name, derived from the filename
                experiment_name = os.path.splitext(os.path.basename(file_path))[0]
                metrics['experiment'] = experiment_name
                all_metrics_data.append(metrics)
                header.update(metrics.keys())
            else:
                print(f"Warning: 'overall_metrics' not found or not a dict in {file_path}. Skipping.")

        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {file_path}. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred while processing {file_path}: {e}")

    if not all_metrics_data:
        print("No valid metrics data was extracted. CSV file will not be created.")
        return

    # Ensure a consistent order for the header, with 'experiment' first
    sorted_header = sorted(list(header))
    if 'experiment' in sorted_header:
        sorted_header.insert(0, sorted_header.pop(sorted_header.index('experiment')))

    # Second pass: write data to CSV
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=sorted_header)
            writer.writeheader()
            writer.writerows(all_metrics_data)
        print(f"Successfully compiled metrics to {output_csv_path}")
    except IOError as e:
        print(f"Error writing to CSV file: {e}")


if __name__ == '__main__':
    compile_metrics_to_csv()
