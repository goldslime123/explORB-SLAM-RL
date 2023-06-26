import re
import uuid
import numpy as np
import ast
import torch


# # Generate a unique number (UUID)
# unique_number = uuid.uuid4()

# # Shorten UUID to the first 7 letters
# shortened_number = str(unique_number)[:7]
# print(shortened_number)

# input_string = "[ 3.9701053  -1.67648514],[-2.24644262 -0.87444116],[ 0.35000026 -3.89999987],[5.50000034 2.70000022],[3.60000031 0.9000002 ]"

# # Replace whitespace with commas
# input_string = input_string.replace(" ", ",")

# # Remove consecutive commas
# input_string = re.sub(r',{2,}', ',', input_string)

# # Remove first and last comma of each bracket
# input_string = re.sub(r'\[(,.*?)(,)\]', r'[\1]', input_string)

# # Remove first comma in each bracket
# output_string = re.sub(r'\[,(.*?)]', r'[\1]', input_string)

# print(output_string)

# # Convert string to list
# output_list = eval(output_string)


# print(output_list)


# centroid_str = '[array([ 1.73647123, -1.0440681 ]), array([3.02500005, 1.45000002]), array([1.65000003, 2.55000004]), array([3.30000005, 4.50000006])]'

# # Remove the word "array" from the string
# centroid_str = centroid_str.replace('array', '')
# # Remove all parentheses from the string
# centroid_str = centroid_str.replace('(', '').replace(')', '')
# # Remove the first and last brackets from the string
# centroid_str = centroid_str[1:-1]

# # Remove all empty spaces in the string
# centroid_str = centroid_str.replace(" ", "")
# print(centroid_str)


# # Remove the surrounding brackets from the string
# centroid_str = centroid_str.strip('[]')

# # Split the string into individual coordinate pairs
# centroid_str = centroid_str.split('],[')

# # Process each pair to create the list of lists
# centroid_str = [list(map(float, pair.split(','))) for pair in centroid_str]

# print(centroid_str)


# import numpy as np
# import csv
# import os


# import csv
# array = np.array([3.18749328, 1.41841822])

# result = np.array2string(array, separator=', ')[1:-1]
# result = result.replace(' ', '')

# print(result)


# empty_list = [[0,0], [0, 0], [0, 0], [0, 0], [0, 0]]
# list_of_lists = [[3.06299043,0.5448482], [1.41894469, 1.80222198], [3.40000005, 3.80000005]]

# for i, sublist in enumerate(list_of_lists):
#     empty_list[i] = sublist

# print(empty_list)

# with open('/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/test/data.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow([result])


# def remove_exponents_from_string(number_string):
#     numbers = number_string.split(',')
#     numbers_without_exponents = []
#     for number in numbers:
#         number = float(number)
#         number_without_exponent = "{0:f}".format(number)
#         numbers_without_exponents.append(number_without_exponent)
#     result = ','.join(numbers_without_exponents)
#     return result

# def string_to_float_list(string):
#     values = string.split(',')
#     float_list = [float(value) for value in values]
#     return float_list


# def string_to_float_list2(string):
#     string_with_comma = string.replace(' ', ', ')
#     values = string_with_comma.strip('[]').split(', ')
#     float_list = [float(value) for value in values]
#     return float_list

# import ast
# def string_to_float_matrix(string_matrix):
#     float_matrix = ast.literal_eval(string_matrix)
#     return float_matrix

# def convert_to_list_of_lists(lst):
#     list_of_lists = [[num] for num in lst]
#     return list_of_lists

# filename = "/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/test/c.csv"
# # Read the CSV file
# with open(filename, 'r') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         robot_position = row[0]
#         robot_orientation = row[1]
#         centroid_record = row[2]
#         info_gain_record = row[3]
#         best_centroid = row[4]
#         # Process the data

# import torch
# # Example usage
# robot_position = string_to_float_list(robot_position)

# robot_orientation = remove_exponents_from_string(robot_orientation)
# robot_orientation = string_to_float_list(robot_orientation)

# centroid_record = string_to_float_matrix(centroid_record)

# info_gain_record = string_to_float_matrix(info_gain_record)
# info_gain_record = convert_to_list_of_lists(info_gain_record)

# best_centroid = string_to_float_list2(best_centroid)

# print("-------------------")
# print(f"Robot Position: {robot_position}")
# print(f"Robot Orientation: {robot_orientation}")
# print(f"Centroid Record: {centroid_record}")
# print(f"Info Gain Record: {info_gain_record}")
# print(f"Best Centroid: {best_centroid}")
# print("-------------------")

# robot_position = [1.6998447315016334,3.52581991835878]
# robot_orientation = [-0.0030015861938741785,0.002416949504063036,0.8137270796538675,0.5812343736311842]
# centroid_record = [4.20577174,-1.40001973],[0.58587992,-0.60072875],[0,0],[0,0],[0,0]
# info_gain_record=[8.421770797635304],[148.53895792332335],[0],[0],[0]


# robot_position = torch.tensor(robot_position, dtype=torch.float32)
# robot_orientation = torch.tensrobot_position = [1.6998447315016334,3.52581991835878]
# robot_orientation = [-0.0030015861938741785,0.002416949504063036,0.8137270796538675,0.5812343736311842]
# centroid_record = [4.20577174,-1.40001973],[0.58587992,-0.60072875],[0,0],[0,0],[0,0]
# info_gain_record=[8.421770797635304],[148.53895792332335],[0],[0],[0]


# robot_position = torch.tensor(robot_position, dtype=torch.float32)
# robot_orientation = torch.tensor(robot_orientation, dtype=torch.float32)
# centroid_record = torch.tensor(centroid_record, dtype=torch.float32)
# info_gain_record = torch.tensor(info_gain_record, dtype=torch.float32)
# robot_state = torch.cat((robot_position, robot_orientation))


# print(robot_state)

# def read_csv(directory):
#     # Read the CSV file
#     df = pd.read_csv(directory,sep=';',header=None)

#     # Extract specified columns
#     df.columns = ['robot_position', 'robot_orientation', 'centroid_record', 'info_gain_record', 'best_centroid']


#     robot_position = df['robot_position'].apply(lambda x: [float(i) for i in x.split(",")])
#     robot_orientation = df['robot_orientation'].apply(lambda x: [float(i) for i in x.split(",")])
#     centroid_record = df['centroid_record'].apply(ast.literal_eval)
#     info_gain_record = df['info_gain_record'].apply(ast.literal_eval)
#     best_centroid = df['best_centroid'].apply(ast.literal_eval)


# def read_csv2(directory):
#     """Reads the input data from a CSV file."""

#     raw_data = pd.read_csv(directory, sep=" ", header=None)

#     robot_position = raw_data[0].apply(
#         lambda x: [float(i) for i in x.split(",")])
#     robot_orientation = raw_data[1].apply(
#         lambda x: [float(i) for i in x.split(",")])
#     centroid_record = raw_data[2].apply(ast.literal_eval)
#     info_gain_record = raw_data[3].apply(ast.literal_eval)
#     best_centroid = raw_data[4].apply(ast.literal_eval)


#     return (
#         robot_position.tolist(),
#         robot_orientation.tolist(),
#         centroid_record.tolist(),
#         info_gain_record.tolist(),
#         best_centroid.tolist()
#     )





# import rospy
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import cv2

# bridge = CvBridge()

# def image_callback(msg):
#     try:
#         # Convert ROS Image message to OpenCV format
#         cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        
#         # Save the image
#         save_path = '/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/rviz/test.png'
#         cv2.imwrite(save_path, cv_image)
#         print("Image saved to", save_path)
#     except Exception as e:
#         print("Error processing image:", str(e))

# rospy.init_node('rviz_image_saver')
# image_topic = '/gridmapper/rectified_map' 
# rospy.Subscriber(image_topic, Image, image_callback)

# rospy.spin()

# import rospy
# from nav_msgs.msg import OccupancyGrid
# import numpy as np
# import cv2

# def occupancy_grid_callback(msg):
#     # Extract occupancy grid data
#     width = msg.info.width
#     height = msg.info.height
#     data = np.array(msg.data).reshape((height, width))

#     # Convert occupancy grid to grayscale image
#     image = (data * 255 / 100).astype(np.uint8)

#     # Save the image
#     save_path = '/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/rviz/test.png'
#     cv2.imwrite(save_path, image)
#     print("Image saved to", save_path)

# rospy.init_node('occupancy_grid_saver')
# occupancy_grid_topic = '/gridmapper/rectified_map' 
# rospy.Subscriber(occupancy_grid_topic, OccupancyGrid, occupancy_grid_callback)

# rospy.spin()


# import os
# import csv

# def store_time(get_time, gazebo_env, repeat_count):
#     csv_folder_path = '/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/RL/csv_time'
#     folder_path = os.path.join(csv_folder_path, gazebo_env)
#     file_name = os.path.join(folder_path, gazebo_env + '_' +
#                              'train_result' + '_' + str(repeat_count) + '.csv')

#     if os.path.exists(folder_path):
#         if os.path.exists(file_name):
#             with open(file_name, 'r', newline='') as file:
#                 reader = csv.reader(file, delimiter=' ')
#                 rows = list(reader)
#             exists = any(str(get_time) in row for row in rows)

#             if not exists:
#                 with open(file_name, 'a', newline='') as file:
#                     writer = csv.writer(file, delimiter=' ')
#                     writer.writerow([get_time])
#         else:
#             with open(file_name, 'w', newline='') as file:
#                 writer = csv.writer(file, delimiter=' ')
#                 writer.writerow([get_time])
#     else:
#         os.makedirs(folder_path, exist_ok=True)
#         with open(file_name, 'w', newline='') as file:
#             writer = csv.writer(file, delimiter=' ')
#             writer.writerow(['Results'])
#             writer.writerow([get_time])

# # Example usage:
# get_time = 40.2  # Replace with your actual value
# gazebo_env = "my_env"  # Replace with your desired folder name
# repeat_count = 1  # Replace with your desired repeat count

# store_time(get_time, gazebo_env, repeat_count)

import subprocess

def save_image():
    # Define the window title of the application
    window_title = 'config.rviz - RViz'

    # Get the window ID of the application window
    result = subprocess.run(['wmctrl', '-l'], capture_output=True, text=True)
    window_id = None
    for line in result.stdout.splitlines():
        if window_title in line:
            window_id = line.split()[0]
            break

    # Capture the screenshot using the import command of the xwd tool
    subprocess.run(['xwd', '-id', window_id, '-out', 'screenshot.xwd'])

    # Convert the captured screenshot to a PNG image using the convert command of the ImageMagick tool
    subprocess.run(['convert', 'screenshot.xwd', 'screenshot.png'])

    # Move the captured screenshot to the desired location
    subprocess.run(['mv', 'screenshot.png', '/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/rviz/aws_house/screenshot.png'])
