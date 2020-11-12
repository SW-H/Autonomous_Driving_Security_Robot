#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ros/ros.h>
#include <tf/transform_datatypes.h>
#include <actionlib_msgs/GoalID.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <move_base_msgs/MoveBaseActionGoal.h>
#include <move_base_msgs/MoveBaseActionResult.h>
#include <robot_local_control/GaitechPose.h>
#include <robot_local_control/GaitechPoseArray.h>
#include <robot_local_control/GaitechBool.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/String.h>


class Navigation
{
    private:
        ros::NodeHandle nh;
        ros::Subscriber sub_goto_cmd, sub_cancel_cmd, sub_result;
        ros::Publisher pub_goto, pub_cancel, pub_complete;
        geometry_msgs::Pose goal_array[100];
        bool goto_state;
        char ss[20];
        int n_goals, cnt_goals;

    public:
        Navigation()
        {   
            sub_goto_cmd = nh.subscribe("/local_control/goto", 1000, &Navigation::GoToCB, this);
            sub_cancel_cmd = nh.subscribe("/local_control/cancel", 1000, &Navigation::CancelCB, this);
            sub_result = nh.subscribe("/move_base/result", 1000, &Navigation::ResultCB, this);
            pub_goto = nh.advertise<geometry_msgs::PoseStamped>("/move_base_simple/goal", 1000);
            pub_cancel = nh.advertise<actionlib_msgs::GoalID>("/move_base/cancel", 1000);
            pub_complete = nh.advertise<robot_local_control::GaitechBool>("/local_control/complete", 1000);

            goto_state = false;
        }

        void GoToCB(const robot_local_control::GaitechPoseArray::ConstPtr& msg)
        {
            if(goto_state == false)
            {
                cnt_goals = 0;
                goto_state = true;

                for(int i=0; i<msg->goals.size(); i++)
                {
                    const robot_local_control::GaitechPose &data = msg->goals[i];
                    geometry_msgs::Quaternion quat;
                    double roll, pitch, yaw;

                    yaw = data.orientation;
                    quat = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);

                    goal_array[i].position.x = data.locationX;
                    goal_array[i].position.y = data.locationY;
                    goal_array[i].orientation = quat;
                    n_goals = i;

                    ROS_INFO("Goal[%d] : [%f, %f, %f]", n_goals, goal_array[i].position.x , goal_array[i].position.y, data.orientation);
                }
                GoToStart();
            }
        }

        void GoToStart()
        {
            geometry_msgs::PoseStamped goal;
            goal.header.frame_id = "map";
            goal.header.stamp = ros::Time::now();
            goal.pose.position.x = goal_array[cnt_goals].position.x;
            goal.pose.position.y = goal_array[cnt_goals].position.y;
            goal.pose.orientation = goal_array[cnt_goals].orientation;

            ROS_INFO("Going to Goal[%d] : [%f, %f]", cnt_goals, goal.pose.position.x, goal.pose.position.y);
            pub_goto.publish(goal);
        }

        void CancelCB(const robot_local_control::GaitechBool& msg)
        {
            if(msg.data == true)
            {
                actionlib_msgs::GoalID cancel_cmd;

                pub_cancel.publish(cancel_cmd);
            }
        }

        void ResultCB(const move_base_msgs::MoveBaseActionResult& msg)
        
        {
            if(goto_state == true)
            {
                if(msg.status.status == 2)
                {
                    ROS_INFO("Goal is canceled");
                    goto_state = false;
                }
                else if(msg.status.status == 3)
                {
                    ROS_INFO("GOAL[%d] is reached", cnt_goals);
                    cnt_goals++;

                    if(cnt_goals > n_goals)
                    {
                        ROS_INFO("GoTo command is completed");
                        goto_state = false;
                        pub_complete.publish(true);
                    }
                    else
                    {
                        ros::Duration(1.0).sleep();
                        GoToStart();
                    }
                }
                else if(msg.status.status == 4)
                {
                    ROS_INFO("Path planning fails");
                    goto_state = false;
                }
            }  
        }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "local_control_navigation");
    Navigation NObject;
    ros::spin();
    return 0;
}
