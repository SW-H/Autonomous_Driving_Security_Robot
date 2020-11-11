#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Quaternion.h>
#include <robot_local_control/GaitechPose.h>
#include <angles/angles.h>

int main(int argc, char ** argv)
{
    // initialize ROS and the node
    ros::init(argc, argv, "robot_pose_publisher");
    ros::NodeHandle nh;
    ros::NodeHandle nh_priv("~");

    // configuring parameters
    std::string map_frame, base_frame;
    double publish_frequency;
    //bool is_stamped;
    ros::Publisher pub, p_pub;

    nh_priv.param<std::string>("map_frame",map_frame,"/map");
    nh_priv.param<std::string>("base_frame",base_frame,"/base_link");
    nh_priv.param<double>("publish_frequency",publish_frequency,10);
    pub = nh.advertise<robot_local_control::GaitechPose>("/current/pose", 1000);
    p_pub = nh.advertise<geometry_msgs::Pose>("robot_pose", 1);

    // create the listener
    tf::TransformListener listener;
    listener.waitForTransform(map_frame, base_frame, ros::Time(), ros::Duration(1.0));

    ros::Rate rate(publish_frequency);
    while (nh.ok())
    {
        tf::StampedTransform transform;
        try
        {
            listener.lookupTransform(map_frame, base_frame, ros::Time(0), transform);

            // construct a pose message
            
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.frame_id = map_frame;
            pose_stamped.header.stamp = ros::Time::now();

            pose_stamped.pose.orientation.x = transform.getRotation().getX();
            pose_stamped.pose.orientation.y = transform.getRotation().getY();
            pose_stamped.pose.orientation.z = transform.getRotation().getZ();
            pose_stamped.pose.orientation.w = transform.getRotation().getW();

            pose_stamped.pose.position.x = transform.getOrigin().getX();
            pose_stamped.pose.position.y = transform.getOrigin().getY();
            pose_stamped.pose.position.z = transform.getOrigin().getZ();

            p_pub.publish(pose_stamped.pose);

            tf::Quaternion quat;
            geometry_msgs::Quaternion orientation_list;

            orientation_list.x = transform.getRotation().getX();
            orientation_list.y = transform.getRotation().getY();
            orientation_list.z = transform.getRotation().getZ();
            orientation_list.w = transform.getRotation().getW();

            tf::quaternionMsgToTF(orientation_list, quat);

            double roll, pitch, yaw;
            tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);
            yaw = angles::normalize_angle_positive(yaw);

            robot_local_control::GaitechPose current_pose;
            current_pose.locationX = transform.getOrigin().getX();
            current_pose.locationY = transform.getOrigin().getY();
            current_pose.orientation = yaw;

            pub.publish(current_pose);
        }
        catch (tf::TransformException &ex)
        {
            // just continue on
        }

        rate.sleep();
    }

    return EXIT_SUCCESS;
}
