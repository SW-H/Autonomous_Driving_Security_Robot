
(cl:in-package :asdf)

(defsystem "robot_local_control-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :std_msgs-msg
)
  :components ((:file "_package")
    (:file "GaitechBool" :depends-on ("_package_GaitechBool"))
    (:file "_package_GaitechBool" :depends-on ("_package"))
    (:file "GaitechPoseArray" :depends-on ("_package_GaitechPoseArray"))
    (:file "_package_GaitechPoseArray" :depends-on ("_package"))
    (:file "GaitechPose" :depends-on ("_package_GaitechPose"))
    (:file "_package_GaitechPose" :depends-on ("_package"))
  ))