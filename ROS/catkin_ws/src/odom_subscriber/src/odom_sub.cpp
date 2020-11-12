#include <nav_msgs/Odometry.h> // Needed for accessing Odometry data

#include <ros/ros.h>           // Needed for creating the node, etc



// This functions get called when a new odometry message gets to the subscriber

// It automatically gets the odometry message as a parameter

// It prints out various parts of the message

void counterCallback(const nav_msgs::Odometry::ConstPtr &msg) {

  // ROS_INFO("%s", msg->header.frame_id.c_str());

  // ROS_INFO("%f", msg->twist.twist.linear.x);

  ROS_INFO("%f", msg->pose.pose.position.x);

}



int main(int argc, char **argv) {

  // Create a node to run the code

  ros::init(argc, argv, "odom_sub_node");

  ros::NodeHandle nh;



  // create a subscriber to the "/odom" topic so we can get the odometry message

  ros::Subscriber sub = nh.subscribe("odom", 1000, counterCallback);

  

  // Run program until manually stopped

  ros::spin();



  return 0;

}
