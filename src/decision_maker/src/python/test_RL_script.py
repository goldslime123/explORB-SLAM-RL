#!/usr/bin/env python3

import subprocess
import time
import rospy
import signal
import os
from variables import repeat_count, explore_time, gazebo_env, algo


class ActiveSLAM:
    def __init__(self, repeat_count, explore_time, decision_maker, gazebo_env):
        self.repeat_count = repeat_count
        self.explore_time = explore_time
        self.decision_maker = decision_maker
        self.gazebo_env = gazebo_env
        self.ctrl_c_pressed = False

        self.gazebo_process = None
        self.decision_maker_process = None

        # Register the handler function for the SIGINT signal
        signal.signal(signal.SIGINT, self.sigint_handler)

    def roslaunch_gazebo(self, env):
        # Launch Gazebo environment
        subprocess.Popen(
            ['roslaunch', 'robot_description', str(env+".launch")])

    def roslaunch_gazebo_decision_maker(self):
        # Launch Gazebo environment and decision-maker component
        self.gazebo_process = subprocess.Popen(
            ['roslaunch', 'robot_description', str(self.gazebo_env+".launch")])
        time.sleep(10)
        self.decision_maker_process = subprocess.Popen(
            ['roslaunch', 'decision_maker', str(self.decision_maker+".launch")])

    def kill_ros_process(self):
        # Kill ROS processes
        subprocess.run(['pkill', '-f', 'gazebo'])
        subprocess.run(['pkill', '-f', 'roscore'])

    def kill_robot_description_node(self):
        # Kill specific ROS nodes related to the robot description
        subprocess.run(['rosnode', 'kill', '/gazebo'])
        if self.gazebo_process is not None:
            self.gazebo_process.kill()
        subprocess.run(['rosnode', 'kill', '/orb_slam2_rgbd'])
        subprocess.run(['rosnode', 'kill', '/robot_1/move_base_node'])
        subprocess.run(['rosnode', 'kill', '/robot_1/robot_state_publisher'])
        subprocess.run(['rosnode', 'kill', '/rosout'])
        subprocess.run(['rosnode', 'kill', '/rviz'])

    def kill_decision_maker_node(self):
        # Kill specific ROS nodes related to the decision-maker component
        subprocess.run(['rosnode', 'kill', '/decision_maker'])
        if self.decision_maker_process is not None:
            self.decision_maker_process.kill()

        subprocess.run(['rosnode', 'kill', '/G_publisher'])
        subprocess.run(['rosnode', 'kill', '/gridmapper'])
        subprocess.run(['rosnode', 'kill', '/octomapper'])
        subprocess.run(['rosnode', 'kill', '/listener_node'])
        subprocess.run(
            ['rosnode', 'kill', '/frontier_detectors/global_detector'])
        subprocess.run(
            ['rosnode', 'kill', '/frontier_detectors/opencv_detector'])
        subprocess.run(['rosnode', 'kill', '/frontier_detectors/filter'])

    def kill_all_process(self):
        # Kill all relevant processes and nodes
        self.kill_robot_description_node()
        self.kill_decision_maker_node()
        self.kill_ros_process()

    def sigint_handler(self, signal, frame):
        # Signal handler for SIGINT (Ctrl+C) signal
        print("Ctrl+C detected! Performing cleanup...")
        self.kill_all_process()
        self.ctrl_c_pressed = True

    def check_new_csv_files(self):
        folder_path = '/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/RL/csv/temp' + \
            '/' + gazebo_env + '/' + str(repeat_count)
        # Get the list of files in the folder
        files = os.listdir(folder_path)

        # Filter the list to only include CSV files
        csv_files = [file for file in files if file.endswith('.csv')]

        # Sort the list of CSV files by modification time
        sorted_files = sorted(csv_files, key=lambda x: os.path.getmtime(
            os.path.join(folder_path, x)))

        # Get the name of the newest CSV file
        newest_csv_file = sorted_files[-1] if sorted_files else None

        return newest_csv_file

    def save_image(self):
        csv_name = self.check_new_csv_files()
        csv_name = str(csv_name)[:7]

        # print(csv_name)
        # Define the window title of the application
        window_title = 'config.rviz - RViz'
        name_not_completed = str(csv_name) + '_not_completed' + '.png'
        file_path_not_completed = '/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/RL/rviz_results/' + gazebo_env + '/' + \
            algo + '/'+str(repeat_count) + '/not_completed'
        save_path_not_completed = os.path.join(
            file_path_not_completed, name_not_completed)

        # print(save_path_not_completed)

        # Create the directory if it doesn't exist
        os.makedirs(file_path_not_completed, exist_ok=True)

        # Get the window ID of the application window
        result = subprocess.run(
            ['wmctrl', '-l'], capture_output=True, text=True)
        window_id = None
        for line in result.stdout.splitlines():
            if window_title in line:
                window_id = line.split()[0]
                break

        # Capture the screenshot using the import command of the xwd tool
        subprocess.run(['xwd', '-id', window_id, '-out', 'screenshot.xwd'])

        # Convert the captured screenshot to a PNG image using the convert command of the ImageMagick tool
        subprocess.run(['convert', 'screenshot.xwd', name_not_completed])

        name_completed = str(csv_name) + '_completed' + '.png'
        file_path_completed = '/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/RL/rviz_results/' + gazebo_env + '/' +\
            algo + '/' + str(repeat_count) + '/completed'
        save_path_completed = os.path.join(file_path_completed, name_completed)
        # print(save_path_completed)

        # Check if the file  exists
        if not os.path.exists(save_path_completed):
            # Move the captured screenshot to the desired location
            subprocess.run(['mv', name_not_completed, save_path_not_completed])
        else:
            print("Save path already exists. Skipping saving the screenshot.")

    def run(self):
        # Main execution method
        rospy.init_node('script', anonymous=False)
        rospy.loginfo(rospy.get_name() + ": Initializing...")

        while not self.ctrl_c_pressed:
            if self.ctrl_c_pressed:
                break

            for x in range(1, self.repeat_count+1):
                print(f'Launching the ROS launch file (attempt {x})...')

                self.roslaunch_gazebo_decision_maker()
                time.sleep(self.explore_time)

                self.save_image()

                self.kill_all_process()
                print(f"Script execution {x} completed.")

            self.ctrl_c_pressed = True
            print(f"Script execution stopped.")


def initialize_active_slam():
    # Initialize the ActiveSLAM object
    decision_maker = 'test_RL_autonomous_agent'
    active_slam = ActiveSLAM(repeat_count, explore_time,
                             decision_maker, gazebo_env)
    return active_slam


if __name__ == '__main__':
    active_slam = initialize_active_slam()
    active_slam.run()
