
(cl:in-package :asdf)

(defsystem "orb_slam2_ros-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :std_msgs-msg
)
  :components ((:file "_package")
    (:file "ORBState" :depends-on ("_package_ORBState"))
    (:file "_package_ORBState" :depends-on ("_package"))
  ))