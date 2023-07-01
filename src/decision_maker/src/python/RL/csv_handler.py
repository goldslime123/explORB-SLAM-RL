import csv
import pandas as pd
import ast
import os


def read_csv(directory):
    df = pd.read_csv(directory, sep=';', header=None)
    df.columns = ['robot_position', 'robot_orientation',
                  'centroid_record', 'info_gain_record', 'best_centroid']

    robot_position = df['robot_position'].apply(ast.literal_eval)
    robot_orientation = df['robot_orientation'].apply(ast.literal_eval)
    centroid_record = df['centroid_record'].apply(ast.literal_eval)
    info_gain_record = df['info_gain_record'].apply(ast.literal_eval)
    best_centroid = df['best_centroid'].apply(ast.literal_eval)

    return (
        robot_position.tolist(),
        robot_orientation.tolist(),
        centroid_record.tolist(),
        info_gain_record.tolist(),
        best_centroid.tolist()
    )


def combine_csv(folder_path, output_file):
    # Create the output folder
    output_folder = os.path.dirname(output_file)
    os.makedirs(output_folder, exist_ok=True)

    merged_data = []
    file_list = os.listdir(folder_path)
    num_files = len(file_list)
    print(f"Number of files in the folder: {num_files}")

    # Iterate over each file in the folder
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)

        # Read the CSV file
        with open(file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            data = list(csv_reader)

            # Append the data to the merged_data
            merged_data.extend(data)

    # Write the merged data to the output CSV file
    with open(output_file, 'w', newline='') as merged_file:
        csv_writer = csv.writer(merged_file)
        csv_writer.writerows(merged_data)

    print("CSV files combined successfully.")


def calculate_average_from_csv(file_path):
    print(file_path)
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        next(reader)  # Skip the header row if present
        column_data = []
        for row in reader:
            if len(row) >= 2:
                column_data.append(float(row[1]))

    if column_data:
        average = sum(column_data) / len(column_data)
        return average
    else:
        return None
