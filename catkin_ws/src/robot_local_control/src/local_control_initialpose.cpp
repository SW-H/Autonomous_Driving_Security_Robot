#include <stdio.h>
#include <stdlib.h>
#include <ros/ros.h>
#include <tf/transform_datatypes.h>
#include <robot_local_control/GaitechPose.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

class Initialpose
{
    private:
        ros::NodeHandle nh;
        ros::Subscriber sub_initialpose_cmd;
        ros::Publisher pub_initialpose;
        float covariance[36] = {0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06853891945200942};

    public:
        Initialpose()
        {
            sub_initialpose_cmd = nh.subscribe("/local_control/initialpose", 1000, &Initialpose::InitialposeCB, this);
            pub_initialpose = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/initialpose", 1000);
        }

        void InitialposeCB(const robot_local_control::GaitechPose& msg)
        {
            geometry_msgs::PoseWithCovarianceStamped initialpose;
            geometry_msgs::Quaternion quat;
            double roll, pitch, yaw;

            yaw = msg.orientation;
            quat = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);

            initialpose.header.frame_id = "map";
            initialpose.pose.pose.position.x = msg.locationX;
            initialpose.pose.pose.position.y = msg.locationY;
            initialpose.pose.pose.orientation = quat;

            for(int i=0; i<36; i++)
            {
                initialpose.pose.covariance[i] = covariance[i];
            }
            

            pub_initialpose.publish(initialpose);
        }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "local_control_initialpose");
    Initialpose IObject;
    ros::spin();
    return 0;
}