; Auto-generated. Do not edit!


(cl:in-package orb_slam2_ros-msg)


;//! \htmlinclude ORBState.msg.html

(cl:defclass <ORBState> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (state
    :reader state
    :initarg :state
    :type cl:fixnum
    :initform 0))
)

(cl:defclass ORBState (<ORBState>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ORBState>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ORBState)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name orb_slam2_ros-msg:<ORBState> is deprecated: use orb_slam2_ros-msg:ORBState instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <ORBState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader orb_slam2_ros-msg:header-val is deprecated.  Use orb_slam2_ros-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'state-val :lambda-list '(m))
(cl:defmethod state-val ((m <ORBState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader orb_slam2_ros-msg:state-val is deprecated.  Use orb_slam2_ros-msg:state instead.")
  (state m))
(cl:defmethod roslisp-msg-protocol:symbol-codes ((msg-type (cl:eql '<ORBState>)))
    "Constants for message type '<ORBState>"
  '((:UNKNOWN . 0)
    (:SYSTEM_NOT_READY . 1)
    (:NO_IMAGES_YET . 2)
    (:NOT_INITIALIZED . 3)
    (:OK . 4)
    (:LOST . 5))
)
(cl:defmethod roslisp-msg-protocol:symbol-codes ((msg-type (cl:eql 'ORBState)))
    "Constants for message type 'ORBState"
  '((:UNKNOWN . 0)
    (:SYSTEM_NOT_READY . 1)
    (:NO_IMAGES_YET . 2)
    (:NOT_INITIALIZED . 3)
    (:OK . 4)
    (:LOST . 5))
)
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ORBState>) ostream)
  "Serializes a message object of type '<ORBState>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'state)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'state)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ORBState>) istream)
  "Deserializes a message object of type '<ORBState>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'state)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'state)) (cl:read-byte istream))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ORBState>)))
  "Returns string type for a message object of type '<ORBState>"
  "orb_slam2_ros/ORBState")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ORBState)))
  "Returns string type for a message object of type 'ORBState"
  "orb_slam2_ros/ORBState")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ORBState>)))
  "Returns md5sum for a message object of type '<ORBState>"
  "22250095e5e0ac7a4ef7c210f7bab3e7")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ORBState)))
  "Returns md5sum for a message object of type 'ORBState"
  "22250095e5e0ac7a4ef7c210f7bab3e7")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ORBState>)))
  "Returns full string definition for message of type '<ORBState>"
  (cl:format cl:nil "Header header~%uint16 state # State from Tracking.h~%# constants for enum-like access~%uint16 UNKNOWN=0~%uint16 SYSTEM_NOT_READY=1~%uint16 NO_IMAGES_YET=2~%uint16 NOT_INITIALIZED=3~%uint16 OK=4~%uint16 LOST=5~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ORBState)))
  "Returns full string definition for message of type 'ORBState"
  (cl:format cl:nil "Header header~%uint16 state # State from Tracking.h~%# constants for enum-like access~%uint16 UNKNOWN=0~%uint16 SYSTEM_NOT_READY=1~%uint16 NO_IMAGES_YET=2~%uint16 NOT_INITIALIZED=3~%uint16 OK=4~%uint16 LOST=5~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ORBState>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     2
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ORBState>))
  "Converts a ROS message object to a list"
  (cl:list 'ORBState
    (cl:cons ':header (header msg))
    (cl:cons ':state (state msg))
))
