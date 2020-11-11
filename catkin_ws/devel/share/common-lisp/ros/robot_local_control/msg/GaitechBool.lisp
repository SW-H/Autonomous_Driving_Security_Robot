; Auto-generated. Do not edit!


(cl:in-package robot_local_control-msg)


;//! \htmlinclude GaitechBool.msg.html

(cl:defclass <GaitechBool> (roslisp-msg-protocol:ros-message)
  ((data
    :reader data
    :initarg :data
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass GaitechBool (<GaitechBool>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GaitechBool>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GaitechBool)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name robot_local_control-msg:<GaitechBool> is deprecated: use robot_local_control-msg:GaitechBool instead.")))

(cl:ensure-generic-function 'data-val :lambda-list '(m))
(cl:defmethod data-val ((m <GaitechBool>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robot_local_control-msg:data-val is deprecated.  Use robot_local_control-msg:data instead.")
  (data m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GaitechBool>) ostream)
  "Serializes a message object of type '<GaitechBool>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'data) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GaitechBool>) istream)
  "Deserializes a message object of type '<GaitechBool>"
    (cl:setf (cl:slot-value msg 'data) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GaitechBool>)))
  "Returns string type for a message object of type '<GaitechBool>"
  "robot_local_control/GaitechBool")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GaitechBool)))
  "Returns string type for a message object of type 'GaitechBool"
  "robot_local_control/GaitechBool")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GaitechBool>)))
  "Returns md5sum for a message object of type '<GaitechBool>"
  "8b94c1b53db61fb6aed406028ad6332a")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GaitechBool)))
  "Returns md5sum for a message object of type 'GaitechBool"
  "8b94c1b53db61fb6aed406028ad6332a")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GaitechBool>)))
  "Returns full string definition for message of type '<GaitechBool>"
  (cl:format cl:nil "bool data~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GaitechBool)))
  "Returns full string definition for message of type 'GaitechBool"
  (cl:format cl:nil "bool data~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GaitechBool>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GaitechBool>))
  "Converts a ROS message object to a list"
  (cl:list 'GaitechBool
    (cl:cons ':data (data msg))
))
