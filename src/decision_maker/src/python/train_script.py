#!/usr/bin/env python3

#import libraries
import subprocess
import time
import rospy
import signal
import csv
import os

class ActiveSLAM:
    def __init__(self, repeat_count, explore_time, decision_maker, gazebo_env):
        self.repeat_count = repeat_count
        self.explore_time = explore_time
        self.decision_maker = decision_maker
        self.gazebo_env = gazebo_env
        self.ctrl_c_pressed = False

        # Register the handler function for the SIGINT signal
        signal.signal(signal.SIGINT, self.sigint_handler)

    def roslaunch_gazebo(self, env):
        # Launch Gazebo environment
        subprocess.Popen(['roslaunch', 'robot_description', str(env+".launch")])

    def roslaunch_gazebo_decision_maker(self):
        # Launch Gazebo environment and decision-maker component
        subprocess.Popen(['roslaunch', 'robot_description', str(self.gazebo_env+".launch")])
        time.sleep(5)
        subprocess.Popen(['roslaunch', 'decision_maker', str(self.decision_maker+".launch")])

    def kill_ros_process(self):
        # Kill ROS processes
        subprocess.run(['pkill', '-f', 'gazebo'])
        subprocess.run(['pkill', '-f', 'roscore'])

    def kill_robot_description_node(self):
        # Kill specific ROS nodes related to the robot description
        subprocess.run(['rosnode', 'kill', '/gazebo'])
        subprocess.run(['rosnode', 'kill', '/orb_slam2_rgbd'])
        subprocess.run(['rosnode', 'kill', '/robot_1/move_base_node'])
        subprocess.run(['rosnode', 'kill', '/robot_1/robot_state_publisher'])
        subprocess.run(['rosnode', 'kill', '/rosout'])
        subprocess.run(['rosnode', 'kill', '/rviz'])

    def kill_decision_maker_node(self):
        # Kill specific ROS nodes related to the decision-maker component
        subprocess.run(['rosnode', 'kill', '/decision_maker'])
        subprocess.run(['rosnode', 'kill', '/G_publisher'])
        subprocess.run(['rosnode', 'kill', '/gridmapper'])
        subprocess.run(['rosnode', 'kill', '/octomapper'])
        subprocess.run(['rosnode', 'kill', '/frontier_detectors/global_detector'])
        subprocess.run(['rosnode', 'kill', '/frontier_detectors/opencv_detector'])
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

    def run(self):
        # Main execution method
        rospy.init_node('script', anonymous=False)
        rospy.loginfo(rospy.get_name() + ": Initializing...")

        while not self.ctrl_c_pressed:
            if self.ctrl_c_pressed:
                break

            for x in range(1, self.repeat_count+1):
                print(f'Launching the ROS launch file (attempt {x})...')
                self.attempt_counter = x

                self.roslaunch_gazebo_decision_maker()
                time.sleep(self.explore_time)

                self.kill_all_process()
                print(f"Script execution {x} completed.")

            self.ctrl_c_pressed = True
            print(f"Script execution stopped.")

def initialize_active_slam():
    # Initialize the ActiveSLAM object
    repeat_count = 2
    explore_time = 50
    decision_maker = 'train_autonomous_agent'
    gazebo_env = 'aws_house'
    active_slam = ActiveSLAM(repeat_count, explore_time, decision_maker, gazebo_env)
    return active_slam


if __name__ == '__main__':
    active_slam = initialize_active_slam()
    active_slam.run()
