ExplORB-SLAM-RL
============
Build on top of ExplORB-SLAM which includes deep reinforcement models
 - Deep Q Network (DQN)
 - Double Deep Q Network (DDQN)
 - Dueling Deep Q Network (Dueling DQN)
 - Dueling Double Deep Q Network (Dueling DDQN)

Original: ExplORB-SLAM
------------

A package for Active visual SLAM using the structure of the underlying pose-graph.

Code used for the paper "ExplORB-SLAM: Active Visual SLAM Exploiting the Pose-graph Topology", accepted for presentation in the Fifth Iberian Robotics Conference (ROBOT 2022).

Tested by jplaced for Ubuntu 20.04, ROS Noetic.

Contact: jplaced@unizar.es, jjgomez@unizar.es, kenji188@gmail.com

Citation
------------

Placed, J. A., Gómez-Rodríguez, J. J., Tardós, J. D., & Castellanos, J. A. (2022). ExplORB-SLAM: Active Visual SLAM Exploiting the Pose-graph Topology. In 2022 Fifth Iberian Robotics Conference (ROBOT).

Dependencies:
------------
- Eigen
- OpenCV
- Python3
  * Numpy
  * Sklearn
  * Numba
  * OpenCV
- Gazebo
- ROS Noetic
  * rviz
  * turtlebot3_teleop
  * gazebo_ros
  * octomap_ros
  * octomap_rviz_plugins
  * move_base

Build Repository
------------
1. Clone repo:
```
Original Repository:
git clone https://github.com/JulioPlaced/ExplORBSLAM.git

Updated Repository:
git clone https://github.com/goldslime123/explORB-SLAM-RL.git
```

2. Compile
```
cd explORB-SLAM-RL/
rm -r build devel
catkin make
```

3. Source the workspace:

  ```
  source devel/setup.bash 
  OR
  source ./setup.bash
  ```

  If sourcing doesn't work properly, try

  ```
  catkin config --no-install
  catkin clean --all
  ```

  and rebuild.


Launch Environments:
------------
  [AWS House](https://github.com/aws-robotics/aws-robomaker-small-house-world):
  ```
  roslaunch robot_description aws_house.launch
  ```
   [AWS Bookstore](https://github.com/aws-robotics/aws-robomaker-small-house-world](https://github.com/aws-robotics/aws-robomaker-bookstore-world)):
  ```
  roslaunch robot_description aws_bookstore.launch
  ```

Launch Decision maker
------------
Autonomous Exploration:
  ```
  roslaunch decision_maker autonomous_agent.launch
  ```
Launch the gazebo environment and decision maker together (Change environment and set repeat/exploration counter in train_script.py):
```
  roslaunch decision_maker train_script_agent.launch
  ```
Test RL modles (Select algo in variables.py):
```
  roslaunch decision_maker test_RL_script_agent.launch
  ```


