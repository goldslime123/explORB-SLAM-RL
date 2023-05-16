#!/usr/bin/env python3

# packages
import subprocess
import time
import rospy
import signal

# Function to relaunch the ROS launch file
def roslaunch_gazebo(env):
    # Start the ROS core (roscore)
    subprocess.Popen(['roscore'])

    # Launch the ROS package with the specified launch filr
    subprocess.Popen(['roslaunch', 'robot_description', str(env+".launch")])

def roslaunch_gazebo_decision_maker(env, decision_maker):
    # Start the ROS core (roscore)
    subprocess.Popen(['roscore'])

    # Launch the ROS package with the specified launch filr
    subprocess.Popen(['roslaunch', 'robot_description', str(env+".launch")])
    subprocess.Popen(['roslaunch', 'decision_maker',str(decision_maker+".launch")])


def kill_ros_process():
    subprocess.run(['pkill', '-f', 'gazebo'])
    subprocess.run(['pkill', '-f', 'roscore'])


def kill_robot_description_node():
    # Execute the 'rosnode kill' command to kill the specified node
    subprocess.run(['rosnode', 'kill', '/gazebo'])
    subprocess.run(['rosnode', 'kill', '/orb_slam2_rgbd'])
    subprocess.run(['rosnode', 'kill', '/robot_1/move_base_node'])
    subprocess.run(['rosnode', 'kill', '/robot_1/robot_state_publisher'])
    subprocess.run(['rosnode', 'kill', '/rosout'])
    subprocess.run(['rosnode', 'kill', '/rviz'])


def kill_decision_maker_node():
    # Execute the 'rosnode kill' command to kill the specified node
    subprocess.run(['rosnode', 'kill', '/decision_maker'])
    subprocess.run(['rosnode', 'kill', '/G_publisher'])
    subprocess.run(['rosnode', 'kill', '/gridmapper'])
    subprocess.run(['rosnode', 'kill', '/octomapper'])
    subprocess.run(['rosnode', 'kill', '/frontier_detectors/global_detector'])
    subprocess.run(['rosnode', 'kill', '/frontier_detectors/opencv_detector'])
    subprocess.run(['rosnode', 'kill', '/frontier_detectors/filter'])


def kill_all_process():
    kill_robot_description_node()
    kill_decision_maker_node()
    kill_ros_process()


# Variable to track whether Ctrl+C was detected
ctrl_c_pressed = False

# Handler function for the SIGINT signal (Ctrl+C)
def sigint_handler(signal, frame):
    print("Ctrl+C detected! Performing cleanup...")
    kill_all_process()


if __name__ == '__main__':
    rospy.init_node('script', anonymous=False)
    rospy.loginfo(rospy.get_name() + ": Initializing...")

    # Specify the number of times to repeat the launch file
    repeat_count = 2
    explore_time = 20     # seconds

    # Fetch all parameters
    rospy.get_param('~repeat_count', repeat_count)
    rospy.get_param('~explore_time', explore_time)

    # Specify the name of the ROS package and launch file
    decision_maker = 'autonomous_agent'
    gazebo_env = 'aws_house'

    # Register the handler function for the SIGINT signal
    signal.signal(signal.SIGINT, sigint_handler)

    # check if ctr; c pressed
    while not ctrl_c_pressed:
        if ctrl_c_pressed==False:
            # Loop to repeat the launch file for the specified number of times
            for i in range(1, repeat_count+1):
                print(f'Launching the ROS launch file (attempt {i})...')
                roslaunch_gazebo_decision_maker(gazebo_env, decision_maker)
                time.sleep(explore_time)
                kill_all_process()
                print(f"Script execution {i} completed.")
        else:
            ctrl_c_pressed = True
            print(f"Script execution {i} stopped.")
            
