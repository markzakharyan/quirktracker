import csv
import argparse
from typing import Union, List, Dict

"""
This is a really crude way of checking the difference between two csv files.
It literally just compares the values in each cell and calculates the percentage difference between them.
It doesn't match up events or anything.
It's just a quick and dirty way of checking the difference between two csv files.
"""

def read_csv(file_path: str) -> Dict[str, List[str]]:
    """
    Read data from a CSV file and return a dictionary of rows indexed by eventID.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - dict: Dictionary of rows indexed by eventID.
    """

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = {row[0]: row for row in reader}
    return data

def calculate_percentage_difference(value1: str, value2: str) -> Union[float, None]:
    """
    Calculate the percentage difference between two values.

    Parameters:
    - value1 (str): First value.
    - value2 (str): Second value.

    Returns:
    - float: Percentage difference between the two values.
    - None: If the values are not valid or cannot be converted to float.
    """
    
    if value1 and value2:
        try:
            num1, num2 = float(value1), float(value2)
            if num1 == 0 and num2 == 0:
                return 0
            denominator = (num1 + num2)
            if denominator == 0:
                return 100
            return 100 * (num2 - num1) / denominator
        except ValueError:
            return None
    return None

def read_csv_grouped_by_eventid(file_path: str) -> Dict[str, List[List[str]]]:
    """
    Read data from a CSV file and group rows by EventID.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - dict: Dictionary of rows grouped by EventID.
    """

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # skip the header

        data = {}

        for row in reader:
            event_id = row[0]
            if event_id not in data:
                data[event_id] = []
            data[event_id].append(row)

    return data

def compare_csvs_by_eventid(file1: str, file2: str, output_file: str) -> None:
    """
    Compare two CSV files based on EventID and write their percentage differences to a new CSV file.

    Parameters:
    - file1 (str): Path to the first CSV file.
    - file2 (str): Path to the second CSV file.
    - output_file (str): Path to the output CSV file where the percentage differences will be written.
    """

    data1 = read_csv_grouped_by_eventid(file1)
    data2 = read_csv_grouped_by_eventid(file2)

    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        
        # Writing the header
        writer.writerow(["EventID", "PID", "Layer", "r[cm]", "phi", "z[cm]", "E[GeV]", "px[GeV]", "py[GeV]", "pz[GeV]"])

        for event_id, rows1 in data1.items():
            rows2 = data2.get(event_id)
            if rows2:
                for row1, row2 in zip(rows1, rows2):
                    diff_row = [calculate_percentage_difference(cell1, cell2) for cell1, cell2 in zip(row1, row2)]
                    writer.writerow([0] + diff_row)

if __name__ == '__main__':
    """
    Main execution of the script.

    Usage:
    - The script expects three arguments: paths to two input CSV files and one output CSV file.
    - It compares the two input CSV files and writes their percentage differences to the output CSV file.

    Example:
    `python checkoutput.py input1.csv input2.csv output.csv`
    """

    parser = argparse.ArgumentParser(description='Compare two CSV files and output their percentage difference.')
    parser.add_argument('file1', type=str, help='First CSV file path.')
    parser.add_argument('file2', type=str, help='Second CSV file path.')
    parser.add_argument('output_file', type=str, help='Output CSV file path.')

    args = parser.parse_args()

    compare_csvs_by_eventid(args.file1, args.file2, args.output_file)
