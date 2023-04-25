ExplORB-SLAM-RL
============

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
  AWS house environment:
  ```
  roslaunch robot_description aws_house.launch
  ```
  AWS bookstore environment:
  ```
  roslaunch robot_description aws_bookstore.launch
  ```
  AWS warehouse environment:
  ```
  roslaunch robot_description aws_warehouse.launch
  ```

Launch Decision maker
------------
  ```
  roslaunch decision_maker autonomous_agent.launch
  ```
