; Auto-generated. Do not edit!


(cl:in-package robot_local_control-msg)


;//! \htmlinclude GaitechPoseArray.msg.html

(cl:defclass <GaitechPoseArray> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (goals
    :reader goals
    :initarg :goals
    :type (cl:vector robot_local_control-msg:GaitechPose)
   :initform (cl:make-array 0 :element-type 'robot_local_control-msg:GaitechPose :initial-element (cl:make-instance 'robot_local_control-msg:GaitechPose))))
)

(cl:defclass GaitechPoseArray (<GaitechPoseArray>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GaitechPoseArray>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GaitechPoseArray)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name robot_local_control-msg:<GaitechPoseArray> is deprecated: use robot_local_control-msg:GaitechPoseArray instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <GaitechPoseArray>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robot_local_control-msg:header-val is deprecated.  Use robot_local_control-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'goals-val :lambda-list '(m))
(cl:defmethod goals-val ((m <GaitechPoseArray>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robot_local_control-msg:goals-val is deprecated.  Use robot_local_control-msg:goals instead.")
  (goals m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GaitechPoseArray>) ostream)
  "Serializes a message object of type '<GaitechPoseArray>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'goals))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'goals))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GaitechPoseArray>) istream)
  "Deserializes a message object of type '<GaitechPoseArray>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'goals) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'goals)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'robot_local_control-msg:GaitechPose))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GaitechPoseArray>)))
  "Returns string type for a message object of type '<GaitechPoseArray>"
  "robot_local_control/GaitechPoseArray")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GaitechPoseArray)))
  "Returns string type for a message object of type 'GaitechPoseArray"
  "robot_local_control/GaitechPoseArray")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GaitechPoseArray>)))
  "Returns md5sum for a message object of type '<GaitechPoseArray>"
  "0457324622f09622155c4325cebe2228")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GaitechPoseArray)))
  "Returns md5sum for a message object of type 'GaitechPoseArray"
  "0457324622f09622155c4325cebe2228")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GaitechPoseArray>)))
  "Returns full string definition for message of type '<GaitechPoseArray>"
  (cl:format cl:nil "Header header~%GaitechPose[] goals~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%# 0: no frame~%# 1: global frame~%string frame_id~%~%================================================================================~%MSG: robot_local_control/GaitechPose~%float64 locationX~%float64 locationY~%float64 orientation~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GaitechPoseArray)))
  "Returns full string definition for message of type 'GaitechPoseArray"
  (cl:format cl:nil "Header header~%GaitechPose[] goals~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%# 0: no frame~%# 1: global frame~%string frame_id~%~%================================================================================~%MSG: robot_local_control/GaitechPose~%float64 locationX~%float64 locationY~%float64 orientation~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GaitechPoseArray>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'goals) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GaitechPoseArray>))
  "Converts a ROS message object to a list"
  (cl:list 'GaitechPoseArray
    (cl:cons ':header (header msg))
    (cl:cons ':goals (goals msg))
))
