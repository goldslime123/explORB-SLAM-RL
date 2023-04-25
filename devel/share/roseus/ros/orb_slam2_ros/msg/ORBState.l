;; Auto-generated. Do not edit!


(when (boundp 'orb_slam2_ros::ORBState)
  (if (not (find-package "ORB_SLAM2_ROS"))
    (make-package "ORB_SLAM2_ROS"))
  (shadow 'ORBState (find-package "ORB_SLAM2_ROS")))
(unless (find-package "ORB_SLAM2_ROS::ORBSTATE")
  (make-package "ORB_SLAM2_ROS::ORBSTATE"))

(in-package "ROS")
;;//! \htmlinclude ORBState.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(intern "*UNKNOWN*" (find-package "ORB_SLAM2_ROS::ORBSTATE"))
(shadow '*UNKNOWN* (find-package "ORB_SLAM2_ROS::ORBSTATE"))
(defconstant orb_slam2_ros::ORBState::*UNKNOWN* 0)
(intern "*SYSTEM_NOT_READY*" (find-package "ORB_SLAM2_ROS::ORBSTATE"))
(shadow '*SYSTEM_NOT_READY* (find-package "ORB_SLAM2_ROS::ORBSTATE"))
(defconstant orb_slam2_ros::ORBState::*SYSTEM_NOT_READY* 1)
(intern "*NO_IMAGES_YET*" (find-package "ORB_SLAM2_ROS::ORBSTATE"))
(shadow '*NO_IMAGES_YET* (find-package "ORB_SLAM2_ROS::ORBSTATE"))
(defconstant orb_slam2_ros::ORBState::*NO_IMAGES_YET* 2)
(intern "*NOT_INITIALIZED*" (find-package "ORB_SLAM2_ROS::ORBSTATE"))
(shadow '*NOT_INITIALIZED* (find-package "ORB_SLAM2_ROS::ORBSTATE"))
(defconstant orb_slam2_ros::ORBState::*NOT_INITIALIZED* 3)
(intern "*OK*" (find-package "ORB_SLAM2_ROS::ORBSTATE"))
(shadow '*OK* (find-package "ORB_SLAM2_ROS::ORBSTATE"))
(defconstant orb_slam2_ros::ORBState::*OK* 4)
(intern "*LOST*" (find-package "ORB_SLAM2_ROS::ORBSTATE"))
(shadow '*LOST* (find-package "ORB_SLAM2_ROS::ORBSTATE"))
(defconstant orb_slam2_ros::ORBState::*LOST* 5)

(defun orb_slam2_ros::ORBState-to-symbol (const)
  (cond
        ((= const 0) 'orb_slam2_ros::ORBState::*UNKNOWN*)
        ((= const 1) 'orb_slam2_ros::ORBState::*SYSTEM_NOT_READY*)
        ((= const 2) 'orb_slam2_ros::ORBState::*NO_IMAGES_YET*)
        ((= const 3) 'orb_slam2_ros::ORBState::*NOT_INITIALIZED*)
        ((= const 4) 'orb_slam2_ros::ORBState::*OK*)
        ((= const 5) 'orb_slam2_ros::ORBState::*LOST*)
        (t nil)))

(defclass orb_slam2_ros::ORBState
  :super ros::object
  :slots (_header _state ))

(defmethod orb_slam2_ros::ORBState
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:state __state) 0)
    )
   (send-super :init)
   (setq _header __header)
   (setq _state (round __state))
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:state
   (&optional __state)
   (if __state (setq _state __state)) _state)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; uint16 _state
    2
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; uint16 _state
       (write-word _state s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; uint16 _state
     (setq _state (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;;
   self)
  )

(setf (get orb_slam2_ros::ORBState :md5sum-) "22250095e5e0ac7a4ef7c210f7bab3e7")
(setf (get orb_slam2_ros::ORBState :datatype-) "orb_slam2_ros/ORBState")
(setf (get orb_slam2_ros::ORBState :definition-)
      "Header header
uint16 state # State from Tracking.h
# constants for enum-like access
uint16 UNKNOWN=0
uint16 SYSTEM_NOT_READY=1
uint16 NO_IMAGES_YET=2
uint16 NOT_INITIALIZED=3
uint16 OK=4
uint16 LOST=5

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

")



(provide :orb_slam2_ros/ORBState "22250095e5e0ac7a4ef7c210f7bab3e7")


