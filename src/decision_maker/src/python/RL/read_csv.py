import os
import csv
import pandas as pd

def read_from_csv(directory):
    raw_data = pd.read_csv(directory,sep=' ',header=None)
    raw_data.columns=['robot_position', 'robot_orientation','centroid_record', 'info_gain_record']

    return raw_data['robot_position'],raw_data['robot_orientation'],raw_data['centroid_record'],raw_data['info_gain_record']

if __name__ == '__main__':
    directory = '/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/csv/aws_house/a.csv'
    read_from_csv(directory)


def check_rows(directory):
    raw_data = pd.read_csv(directory, sep=' ', header=None)
    raw_data.columns = ['robot_position', 'robot_orientation', 'centroid_record', 'info_gain_record']

    for index, row in raw_data.iterrows():
        # Perform checks on each row
        if row['robot_position'] == 'some_value':
            print("Found a row with a specific value in the 'robot_position' column.")
        elif row['robot_orientation'] > 0:
            print("Found a row with a positive value in the 'robot_orientation' column.")

        # Add more checks as needed

if __name__ == '__main__':
    directory = '/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/csv/aws_house/a.csv'
    check_rows(directory)