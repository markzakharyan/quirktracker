import csv
import argparse
from typing import Union

def read_csv(file_path: str) -> list:
    """
    Read data from a CSV file.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - list: List of rows, where each row is a list of values.
    """

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
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


def compare_csvs(file1: str, file2: str, output_file: str) -> None:
    """
    Compare two CSV files cell by cell, calculating their percentage differences.
    Writes the percentage differences to a new CSV file.

    Parameters:
    - file1 (str): Path to the first CSV file.
    - file2 (str): Path to the second CSV file.
    - output_file (str): Path to the output CSV file where the percentage differences will be written.

    Notes:
    - The two input CSV files should have the same dimensions.
    - The output CSV file will have the same dimensions as the input files, with the cells containing the percentage differences.
    """

    data1 = read_csv(file1)
    data2 = read_csv(file2)

    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(data1[0])

        for row1, row2 in zip(data1[1:], data2[1:]):
            diff_row = [calculate_percentage_difference(cell1, cell2) for cell1, cell2 in zip(row1, row2)]
            writer.writerow(diff_row)

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

    compare_csvs(args.file1, args.file2, args.output_file)
