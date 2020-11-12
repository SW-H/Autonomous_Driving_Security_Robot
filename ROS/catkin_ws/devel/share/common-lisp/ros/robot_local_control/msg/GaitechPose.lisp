; Auto-generated. Do not edit!


(cl:in-package robot_local_control-msg)


;//! \htmlinclude GaitechPose.msg.html

(cl:defclass <GaitechPose> (roslisp-msg-protocol:ros-message)
  ((locationX
    :reader locationX
    :initarg :locationX
    :type cl:float
    :initform 0.0)
   (locationY
    :reader locationY
    :initarg :locationY
    :type cl:float
    :initform 0.0)
   (orientation
    :reader orientation
    :initarg :orientation
    :type cl:float
    :initform 0.0))
)

(cl:defclass GaitechPose (<GaitechPose>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GaitechPose>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GaitechPose)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name robot_local_control-msg:<GaitechPose> is deprecated: use robot_local_control-msg:GaitechPose instead.")))

(cl:ensure-generic-function 'locationX-val :lambda-list '(m))
(cl:defmethod locationX-val ((m <GaitechPose>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robot_local_control-msg:locationX-val is deprecated.  Use robot_local_control-msg:locationX instead.")
  (locationX m))

(cl:ensure-generic-function 'locationY-val :lambda-list '(m))
(cl:defmethod locationY-val ((m <GaitechPose>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robot_local_control-msg:locationY-val is deprecated.  Use robot_local_control-msg:locationY instead.")
  (locationY m))

(cl:ensure-generic-function 'orientation-val :lambda-list '(m))
(cl:defmethod orientation-val ((m <GaitechPose>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robot_local_control-msg:orientation-val is deprecated.  Use robot_local_control-msg:orientation instead.")
  (orientation m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GaitechPose>) ostream)
  "Serializes a message object of type '<GaitechPose>"
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'locationX))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'locationY))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'orientation))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GaitechPose>) istream)
  "Deserializes a message object of type '<GaitechPose>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'locationX) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'locationY) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'orientation) (roslisp-utils:decode-double-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GaitechPose>)))
  "Returns string type for a message object of type '<GaitechPose>"
  "robot_local_control/GaitechPose")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GaitechPose)))
  "Returns string type for a message object of type 'GaitechPose"
  "robot_local_control/GaitechPose")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GaitechPose>)))
  "Returns md5sum for a message object of type '<GaitechPose>"
  "b809aeffe83b44060751544686db7c54")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GaitechPose)))
  "Returns md5sum for a message object of type 'GaitechPose"
  "b809aeffe83b44060751544686db7c54")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GaitechPose>)))
  "Returns full string definition for message of type '<GaitechPose>"
  (cl:format cl:nil "float64 locationX~%float64 locationY~%float64 orientation~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GaitechPose)))
  "Returns full string definition for message of type 'GaitechPose"
  (cl:format cl:nil "float64 locationX~%float64 locationY~%float64 orientation~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GaitechPose>))
  (cl:+ 0
     8
     8
     8
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GaitechPose>))
  "Converts a ROS message object to a list"
  (cl:list 'GaitechPose
    (cl:cons ':locationX (locationX msg))
    (cl:cons ':locationY (locationY msg))
    (cl:cons ':orientation (orientation msg))
))
