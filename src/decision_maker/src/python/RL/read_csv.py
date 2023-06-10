import os
import csv
import pandas as pd

def read_from_csv(directory):
    raw_data = pd.read_csv(directory,sep=' ',header=None)
    raw_data.columns=['robot_position', 'robot_orientation','centroid_record', 'info_gain_record']
    
    # Print DataFrame without index 
    blankIndex=[''] * len(raw_data)
    raw_data.index=blankIndex
    print(raw_data['robot_position'])

    return raw_data['robot_position'],raw_data['robot_orientation'],raw_data['centroid_record'],raw_data['info_gain_record']

if __name__ == '__main__':
    directory = '/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/csv/aws_house/a.csv'
    read_from_csv(directory)