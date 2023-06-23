#!/usr/bin/env python3

# jplaced
# 2022, Universidad de Zaragoza

# This node receives target exploration goals, which are the filtered frontier
# points published by the filter node, and commands the robots accordingly. The
# controller node commands the robot through the move_base_node.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Include modules~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# libraries
import os
import csv
import uuid
import re
from variables import gazebo_env,output_size,repeat_count,no_frontier_counter
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import rospy
import tf
import heapq
import numpy as np
import dynamic_reconfigure.client
from numpy import array
from copy import deepcopy
from scipy.spatial.transform import Rotation
from frontier_detector.msg import PointArray
from nav_msgs.msg import OccupancyGrid
"""
from sensor_msgs.msg import PointCloud2
"""
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Pose
from visualization_msgs.msg import MarkerArray
from actionlib_msgs.msg import GoalID
from orb_slam2_ros.msg import ORBState

from Functions import waitEnterKey, quaternion2euler, cellInformation_NUMBA, cellInformation, yawBtw2Points
from Robot import Robot
from Map import Map
from WeightedPoseGraph import WeightedPoseGraph
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Callbacks~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
gridMap_data_ = OccupancyGrid()
frontiers_ = []
mapPoints_ = []
vertices_ = []
edges_ = []

is_lost_ = False
is_relocalizing_ = False
goal_cancel_pub_ = rospy.Publisher(
    '/robot_1/move_base/cancel', GoalID, queue_size=10)

def does_row_exist(file_name, row_data):
    with open(file_name, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if row == row_data:
                return True
    return False


def format_centroid_record(centroid_record, size):
    centroid_empty_list = [[0, 0] for _ in range(size)]
    for i, sublist in enumerate(centroid_record):
        centroid_empty_list[i] = sublist

    centroid_empty_list = str(centroid_empty_list)
    centroid_empty_list = centroid_empty_list.replace('array', '')
    centroid_empty_list = centroid_empty_list.replace('(', '').replace(')', '')
    centroid_empty_list = centroid_empty_list[1:-1]
    centroid_empty_list = centroid_empty_list.replace(" ", "")
    centroid_empty_list = centroid_empty_list.strip('[]')
    centroid_empty_list = centroid_empty_list.split('],[')

    centroid_empty_list = [list(map(float, pair.split(',')))
                           for pair in centroid_empty_list]

    return centroid_empty_list


def format_info_gain_record(info_gain_record, size):
    info_empty_list = [[0] for _ in range(size)]
    for i, sublist in enumerate(info_gain_record):
        info_empty_list[i] = sublist

    info_empty_list = str(info_empty_list)
    # Split the string by commas
    elements = info_empty_list.split(",")

    # Process each element individually
    output_list = []
    for element in elements:
        element = element.strip()  # Remove leading and trailing whitespace
        
        if element != "[0]":  # Add brackets if element is not [0]
            element = f"[{element.strip('[]')}]"
        
        output_list.append(element)

    # Join the elements back into a string
    output_string = ", ".join(output_list)
    output_string = '[' + output_string + ']'

    return output_string


def format_list(string):
    string = str(string)
    # Remove the first and last brackets from the string
    string = string[1:-1]

    # Remove empty spaces from the front
    string = string.lstrip()

    string = string.rstrip()

    string = string.replace(" ", ",")

    string = re.sub(r',,', ',', string)

    string = '[' + string + ']'

    return string



# Generate a unique number (UUID)
unique_number = uuid.uuid4()
shortened_number = str(unique_number)[:7]
def store_csv(robot_position, robot_orientation, centroid_record, info_gain_record, best_centroid):
    csv_folder_path = '/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/csv'
    folder_path = csv_folder_path + '/' + gazebo_env + '/'+ str(repeat_count)
    file_name = folder_path  +'/'+ str(shortened_number) + '.csv'

    robot_position = format_list(robot_position)
    robot_orientation = format_list(robot_orientation)
    centroid_record = format_centroid_record(centroid_record, output_size)
    info_gain_record = format_info_gain_record(info_gain_record, output_size)
    best_centroid = format_list(best_centroid)

    if os.path.exists(folder_path):
        print("The folder exists.")
        if os.path.exists(file_name):
            print(f"The file '{file_name}' exists.")
            with open(file_name, 'a', newline='') as file:
                writer = csv.writer(file, delimiter=';')
                writer.writerow(
                    [robot_position, robot_orientation, centroid_record, info_gain_record, best_centroid])

        else:
            print(f"The file '{file_name}' does not exist. Creating file.")
            with open(file_name, 'w', newline='') as file:
                writer = csv.writer(file, delimiter=';')
                writer.writerow(
                    [robot_position, robot_orientation, centroid_record, info_gain_record, best_centroid])
    else:
        print("The folder does not exist.")
        os.makedirs(folder_path, exist_ok=True)
        print(f"Folder '{folder_path}' created successfully.")


def mapPointsCallBack(data):
    global mapPoints_

    mapPoints_ = []
    temp = []
    for i in range(0, len(data.data)):
        if data.data[i] == -1.0:
            mapPoints_.append(temp)
            temp = []
        else:
            temp.append(data.data[i])


def vertexCallBack(data):
    global vertices_
    n = data.layout.dim[0].stride

    vertices_ = []
    for i in range(0, len(data.data), n):
        vertices_.append(data.data[i:i + n])


def edgesCallBack(data):
    global edges_
    n = data.layout.dim[0].stride

    edges_ = []
    for i in range(0, len(data.data), n):
        edges_.append(data.data[i:i + n])


def frontiersCallBack(data):
    global frontiers_
    frontiers_ = []
    for point in data.points:
        frontiers_.append(array([point.x, point.y]))


def mapCallBack(data):
    global gridMap_data_
    gridMap_data_ = data


def statusCallBack(data):
    """
    UNKNOWN=0, SYSTEM_NOT_READY=1, NO_IMAGES_YET=2, NOT_INITIALIZED=3, OK=4, LOST=5
    """
    global is_lost_, is_relocalizing_
    if data.state == 4:
        is_lost_ = False
        if is_relocalizing_:  # Stop trying to re localize if already OK
            msg = GoalID()
            goal_cancel_pub_.publish(msg)
            is_relocalizing_ = False
            rospy.loginfo(rospy.get_name() +
                          ': ORB-SLAM re localized successfully.')
    # If lost, cancel current goal and send best re localization goal
    elif data.state == 5 and not is_relocalizing_:
        # Empty stamp, empty ID -> cancels ALL goals.
        # https://wiki.ros.org/actionlib/DetailedDescription
        msg = GoalID()
        goal_cancel_pub_.publish(msg)
        is_lost_ = True
        rospy.logwarn_throttle(1, rospy.get_name() + ': ORB-SLAM status is LOST. Robot stopped.'
                                                     ' Sending robot to best re localization pose.')
    elif data.state == 0:  # Stop robot
        msg = GoalID()
        goal_cancel_pub_.publish(msg)
        rospy.logwarn_throttle(1, rospy.get_name() +
                               ': ORB-SLAM status is UNKNOWN. Robot stopped.')

counter = 0
store_result =[]
def log_warn_throttled(message):
    global counter
    counter += 1
    rospy.logwarn_throttle(0.5, rospy.get_name ()+ message)
    rospy.loginfo("Log warn counter: {}".format(counter))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Node~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def node():
    global frontiers_, mapPoints_, vertices_, edges_, gridMap_data_, is_relocalizing_
    rospy.init_node('decision_maker', anonymous=False)
    rospy.loginfo(rospy.get_name() + ": Initializing...")

    # Fetch all parameters
    map_topic = rospy.get_param('~map_topic', '/map')
    frontiers_topic = rospy.get_param('~frontiers_topic', '/filtered_points')
    n_robots = rospy.get_param('~n_robots', 1)
    namespace = rospy.get_param('~namespace', 'robot_')
    rate_hz = rospy.get_param('~rate', 100)
    delay_after_assignment = rospy.get_param('~delay_after_assignment', 0.1)
    show_debug_path = rospy.get_param('~show_debug_path', False)
    exploring_time = rospy.get_param('~max_exploring_time', 9000)
    use_gpu = rospy.get_param('~enable_gpu_comp', True)
    camera_type = rospy.get_param('~camera_type', 'rgbd')

    rate = rospy.Rate(rate_hz)
    tf_listener = tf.TransformListener()

    rospy.Subscriber(map_topic, OccupancyGrid, mapCallBack)
    rospy.Subscriber(frontiers_topic, PointArray, frontiersCallBack)
    rospy.Subscriber("/orb_slam2_" + camera_type +
                     "/info/state", ORBState, statusCallBack)

    rospy.Subscriber("/orb_slam2_" + camera_type +
                     "/vertex_list", Float64MultiArray, vertexCallBack)
    rospy.Subscriber("/orb_slam2_" + camera_type +
                     "/edge_list", Float64MultiArray, edgesCallBack)
    rospy.Subscriber("/orb_slam2_" + camera_type + "/point_list",
                     Float64MultiArray, mapPointsCallBack)

    if show_debug_path:
        marker_hallucinated_path_pub_ = rospy.Publisher(
            'marker_hallucinated_path', MarkerArray, queue_size=10)
        """
        point_cloud2_map_pub_ = rospy.Publisher("marker_points_frustum", PointCloud2, queue_size=1)
        """
        marker_hallucinated_graph_pub_ = rospy.Publisher(
            'marker_hallucinated_graph', MarkerArray, queue_size=10)

    # Wait if map is not received yet
    while len(gridMap_data_.data) < 1:
        pass
    rospy.loginfo(rospy.get_name() + ": Controller received map.")

    # Robot
    robot_name = namespace + str(n_robots)
    robot_ = Robot(robot_name)

    # ORB-SLAM map
    map_ = Map()
    
    #temporal cond
    t_0 = rospy.get_time()

    
    ig_changer = 0

    rospy.loginfo(rospy.get_name() + ": Initialized.")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    while not rospy.is_shutdown():

        # Check temporal stopping criterion
        t_f = rospy.get_time() - t_0  # Get ROS time in seconds
        # rospy.loginfo("Current time ROS: %f ", t_f)

        if t_f >= exploring_time:
            robot_.cancelGoal()
            waitEnterKey()

        # Get tf from camera link to base frame
        cond = 0
        while cond == 0:
            try:
                (t_camera_base, q_camera_base) = tf_listener.lookupTransform("base_footprint", "camera_link_optical",
                                                                             rospy.Time(0))
                map_.setCameraBaseTf(t_camera_base, q_camera_base)
                cond = 1
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.logerr(rospy.get_name() +
                             ": Could not get TF from base to camera link.")
                cond = 0

        if not is_lost_:  # ORB-SLAM OK
            centroids = deepcopy(frontiers_)
            n_centroids = len(centroids)
            

            if n_centroids <= 0:
                # rospy.logwarn_throttle(
                #     0.5, rospy.get_name() + ": No frontiers.")
                log_warn_throttled("No frontiers.")

                if counter >= no_frontier_counter:
                    #get current ros time
                    current_time = rospy.Time.now()
                    store_result.append(current_time.to_sec())
                    # print("store_result:", store_result)
                    print("Completed ROS time:", store_result[0])
                
                if ig_changer < 10:
                    ig_changer += 1
                client = dynamic_reconfigure.client.Client(
                    "/frontier_detectors/filter")
                if ig_changer > 10 and client.get_configuration(timeout=1)["ig_threshold"] != 0.1:
                    client.update_configuration({"ig_threshold": 0.1})
            else:
                # Get SLAM graph
                map_.setNodes(vertices_)
                map_.setEdges(edges_)
                map_.setMapPoints(mapPoints_)

                nodes, edges = map_.getNodesEdges()

                # Build nx graph
                G = WeightedPoseGraph(nodes, edges, 'd_opt')

                # If no nodes (starting step) build graph with one edge at origin.
                n = float(G.getNNodes())
                m = float(G.getNEdges())
                if n < 1:
                    G.graph.add_node(int(0), translation=[
                                     0, 0, 0], orientation=[0, 0, 0, 0])

                info_gain = []
                closer_goal = False
                single_goal = False

                # If only one frontier no need to evaluate anything. Select that frontier
                if n_centroids == 1:
                    rospy.logwarn(rospy.get_name() +
                                  ": Only one frontier detected. Selecting it.")
                    single_goal = True
                # If no edges (starting step) do not evaluate D-opt. Select random frontier
                elif m < 1:
                    rospy.logwarn(rospy.get_name(
                    ) + ": Graph not started yet, m < 1. Selecting goal +=[0.1,0.1].")
                    closer_goal = True
                else:  # Otherwise
                    rospy.loginfo(
                        rospy.get_name() + ": Computing information gain of every frontier candidate.")
                    for ip in range(0, n_centroids):
                        # Get frontier goal
                        p_frontier = np.array(
                            [centroids[ip][0], centroids[ip][1]])

                        # Compute hallucinated pose graph
                        if show_debug_path:
                            if use_gpu:
                                seen_cells_pct = cellInformation_NUMBA(np.array(gridMap_data_.data),
                                                                       gridMap_data_.info.resolution,
                                                                       gridMap_data_.info.width,
                                                                       gridMap_data_.info.origin.position.x,
                                                                       gridMap_data_.info.origin.position.y,
                                                                       p_frontier[0], p_frontier[1], 0.5)
                            else:
                                seen_cells_pct = cellInformation(
                                    gridMap_data_, p_frontier, 0.5)

                            hallucinated_path, G_frontier = G.hallucinateGraph(robot_, map_, seen_cells_pct, p_frontier,
                                                                               True)
                            marker_hallucinated_path_pub_.publish(
                                hallucinated_path)
                            marker_hallucinated_graph_pub_.publish(
                                G_frontier.getGraphAsMarkerArray(color=False))
                            # waitEnterKey()
                            """
                            visualizing_pts = map_.getMapPointsAsROSPointCloud2()
                            pts = map_.frustumCulling(robot_.getPoseAsGeometryMsg())
                            visualizing_pts = map_.getMapPointsAsROSPointCloud2("map", pts)
                            point_cloud2_map_pub_.publish(visualizing_pts)
                            waitEnterKey()
                            """
                        else:
                            G_frontier = G.hallucinateGraph(
                                robot_, map_, p_frontier, False)

                        # Compute no. of spanning trees <=> D-opt(FIM)
                        n_frontier = float(G_frontier.getNNodes())
                        if n_frontier > 0:
                            L_anchored = G_frontier.computeAnchoredL()
                            _, t = np.linalg.slogdet(L_anchored.todense())
                            n_spanning_trees = n_frontier ** (
                                1 / n_frontier) * np.exp(t / n_frontier)
                            info_gain.append(n_spanning_trees)

                # Goal sender
                if robot_.getState() == 1:
                    rospy.logwarn(rospy.get_name() +
                                  ": Robot is not available.")
                elif closer_goal:
                    robot_.sendGoal(robot_.getPosition() + [0.1, 0.1], True)
                elif single_goal:
                    rospy.loginfo(rospy.get_name(
                    ) + ": " + format(robot_name) + " assigned to " + format(centroids[0]))
                    robot_.sendGoal(centroids[0], True)

                elif len(info_gain) > 0:
                    # Select next best frontier
                    info_gain_record = []
                    centroid_record = []

                    for ip in range(0, len(centroids)):
                        info_gain_record.append(info_gain[ip])
                        centroid_record.append(centroids[ip])

                    winner_id = info_gain_record.index(
                        np.max(info_gain_record))
                    info_centroid_record = dict(
                        zip(info_gain_record, centroid_record))

                    robot_pose = robot_.getPoseAsGeometryMsg()
                    print("Robot Pose: \n", robot_pose)

                    rospy.loginfo(rospy.get_name() +
                                  ": Centroids: \n" + format(centroid_record))
                    rospy.loginfo(
                        rospy.get_name() + ": Information gain: \n" + format(info_gain_record))

                    rospy.loginfo(
                        rospy.get_name() + ": Info gain/Centroid: \n" + format(info_centroid_record))
                    
                    rospy.loginfo(rospy.get_name() + ": " + format(robot_name) + " assigned to centroid " +
                                  format(centroid_record[winner_id]))

                    # Get robot's current pose
                    robot_position = robot_.getPose()[0]
                    robot_orientation = robot_.getPose()[1]

                    # store csv
                    store_csv(robot_position, robot_orientation,
                              centroid_record, info_gain_record, centroid_record[winner_id])

                    # Send goal to robot
                    initial_plan_position = robot_.getPosition()
                    robot_.sendGoal(centroid_record[winner_id], True)
                    # robot_.sendGoal([2 ,2], True)

                    # If plan fails near to starting position, send new goal to the next best frontier
                    if robot_.getState() != 3:
                        euclidean_d = np.linalg.norm(
                            robot_.getPosition() - initial_plan_position)
                        if euclidean_d <= 2.0:
                            new_goal = 2
                            while robot_.getState() != 3 and new_goal <= len(info_gain_record):
                                second_max = heapq.nlargest(
                                    new_goal, info_gain_record)[1]
                                winner_id = info_gain_record.index(second_max)
                                rospy.logwarn(rospy.get_name() + ": Goal aborted near previous pose (eucl = " +
                                              str(euclidean_d) + "). Sending new goal to: " +
                                              str(centroid_record[winner_id]))
                                robot_.sendGoal(
                                    centroid_record[winner_id], True)
                                new_goal = new_goal + 1

                        else:
                            rospy.logwarn(rospy.get_name() + ": Goal aborted away from previous pose (eucl = " +
                                          format(euclidean_d) + "). Recomputing.")

        else:  # ORB-SLAM lost
            is_relocalizing_ = True
            while is_lost_:
                best_reloc_poses = map_.getBestRelocPoses(
                    robot_.getPoseAsGeometryMsg())
                rospy.logwarn(rospy.get_name(
                ) + ": ORB-SLAM lost. Sending robot to best re localization pose.")
                for reloc_poses in best_reloc_poses:
                    _, _, reloc_yaw = quaternion2euler(reloc_poses.orientation.w, reloc_poses.orientation.x,
                                                       reloc_poses.orientation.y, reloc_poses.orientation.z)
                    reloc_position = [
                        reloc_poses.position.x, reloc_poses.position.y]
                    rospy.loginfo(rospy.get_name() + ": " + format(robot_name) + " assigned to [" +
                                  format(reloc_position) + ", " + format(reloc_yaw * 180 / 3.14159) + "]")
                    robot_.sendGoalAsPose(reloc_poses, True)

        # Wait delay after assignment
        rospy.sleep(delay_after_assignment)

        rate.sleep()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Main~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == '__main__':
    try:
        node()

    except rospy.ROSInterruptException:
        pass
