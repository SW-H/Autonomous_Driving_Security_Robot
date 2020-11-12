#include <ros/ros.h>
#include <ros/console.h>
#include <tf/transform_datatypes.h>
#include <std_msgs/String.h>
#include <std_msgs/Bool.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>

// remote contol command = forward, backward, right, left

#define DIFF_ABS(X,Y) ((X)>(Y)? (X)-(Y) : (Y)-(X))

class Remote
{
    private:
        ros::NodeHandle nh;
        ros::Subscriber sub_remote, sub_odom, sub_all_stop;
        ros::Publisher pub_vel;
        bool remote_state;
        std::string cmd_remote;
        geometry_msgs::Pose last_pose, cur_pose;
        geometry_msgs::Twist cmd_vel;
        float distance,rotation;
        double last_yaw, cur_yaw, move_meter, move_rad;

    public:
        Remote()
        {
            ros::NodeHandle pnh("~");
            sub_remote = nh.subscribe("/remote_control", 1000, &Remote::RemoteCB, this);
            sub_odom = nh.subscribe("/odom", 1000, &Remote::OdomCB, this);
            //sub_all_stop = nh.subscribe("/all_stop", 1000, &Remote::AllStopCB, this);
            pub_vel = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 1000);
            pnh.param("move_meter", move_meter, 0.2);
            pnh.param("move_rad", move_rad, 0.314);
        }

        void RemoteCB(const std_msgs::String& msg)
        {
            last_pose.position.x = cur_pose.position.x;
            last_pose.position.y = cur_pose.position.y;
            last_pose.orientation = cur_pose.orientation;
            last_yaw = tf::getYaw(last_pose.orientation);

            cmd_remote = msg.data;
            remote_state = true;

            ROS_INFO("Remote Command : [%s]", cmd_remote.c_str());
        }

        void OdomCB(const nav_msgs::Odometry& msg)
        {
            cur_pose.position.x = msg.pose.pose.position.x;
            cur_pose.position.y = msg.pose.pose.position.y;
            cur_pose.orientation = msg.pose.pose.orientation;

            if(remote_state == true)
            {
                cmd_vel.linear.x = 0.0;
                cmd_vel.angular.z = 0.0;

                distance = sqrt(powf(cur_pose.position.x - last_pose.position.x, 2.0)
                            + powf(cur_pose.position.y - last_pose.position.y, 2.0));
                cur_yaw = tf::getYaw(cur_pose.orientation);
                rotation = DIFF_ABS(last_yaw, cur_yaw);

                if(cmd_remote == "forward" && distance < move_meter)
                    cmd_vel.linear.x = 0.1;
                else if(cmd_remote == "backward" && distance < move_meter)
                    cmd_vel.linear.x = -0.1;
                else if(cmd_remote == "right" && rotation < move_rad)
                    cmd_vel.angular.z = -0.2;
                else if(cmd_remote == "left" && rotation < move_rad)
                    cmd_vel.angular.z = 0.2;
                else
                {
                    cmd_vel.linear.x = 0.0;
                    cmd_vel.angular.z = 0.0;
                    remote_state = false;
                }
                    
                pub_vel.publish(cmd_vel);
            }
        }

        /*void AllStopCB(const std_msgs::Bool& msg)
        {
            if(msg.data == true)
                remote_state = false;
        }*/
};


int main(int argc, char **argv)
{
    ros::init(argc, argv, "remote_control");
    Remote RObject;
    ros::spin();
    return 0;
}
