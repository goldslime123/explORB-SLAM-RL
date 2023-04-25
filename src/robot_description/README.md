ExplORB-SLAM with Reinforcement Learning
============

A package for Active visual SLAM using the structure of the underlying pose-graph.

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

Building
------------
1. Clone repo:
```
https://github.com/goldslime123/explORB-SLAM-RL.git
```

2. Build repo:
```
cd Explore_ORBSLAM/
catkin make
```

3. Remember to source the ExplORBSLAM workspace:

  ```
  source devel/setup.bash
  ```

  If sourcing doesn't work properly, try

  ```
  catkin config --no-install
  catkin clean --all
  ```

  and rebuild.

Running
------------
### Launch environments

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
  Turtlebot3 environment:
  ```
  roslaunch robot_description turtlebot3_house.launch
  ```
### Launch decision maker (exploration)
  ```
  roslaunch decision_maker autonomous_agent.launch
  ```
