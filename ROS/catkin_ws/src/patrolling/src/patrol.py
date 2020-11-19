#!/usr/bin/env python

import rospy
import actionlib
import tf
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import requests
from sound_play.msg import SoundRequest
from sound_play.libsoundplay import SoundClient
import rospy
import socket
# the list of points to patrol

def sound_play_func(ment):
        soundhandle = SoundClient()
        rospy.sleep(1)
        voice = 'voice_kal_diphone'
        soundhandle.say(ment, voice)
        rospy.sleep(1)

waypoints = [
    ['one', (-6.68705231286,-2.58100135396,0.00)],
    ['two', (-10.5950424893,-1.37614052912,0.00)],
    ['three', (-10.6175562333,-5.02315051105,0.00)],
    ['four',(-8.75326920292,-4.81857861109,0.00)]
]

class Patrol:

    def __init__(self):
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()

    def set_goal_to_point(self, point):

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = point[0]
        goal.target_pose.pose.position.y = point[1]
        quaternion = tf.transformations.quaternion_from_euler(0.0, 0.0, point[2])
        goal.target_pose.pose.orientation.x = quaternion[0]
        goal.target_pose.pose.orientation.y = quaternion[1]
        goal.target_pose.pose.orientation.z = quaternion[2]
        goal.target_pose.pose.orientation.w = quaternion[3]

        self.client.send_goal(goal)
        wait = self.client.wait_for_result()
        if not wait:
            rospy.logerr("Action server not available!")
            rospy.signal_shutdown("Action server not available!")
        else:
            return self.client.get_result()


if __name__ == '__main__':
    rospy.init_node('')#patrolling
    server_socket=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind(('172.22.77.52',7010)) # PC2 windows

    try:
        p = Patrol()
        while not rospy.is_shutdown():
            for i, w in enumerate(waypoints):
		data,addr = server_socket.recvfrom(200)	
	    	if data.decode()=="":
			rospy.loginfo("Sending waypoint %d - %s", i, w[0])
			rospy.init_node('')
       	        	p.set_goal_to_point(w[1])
		else:
		   #sound_play_func()
 		   print "received data:", data.decode()
		   rospy.loginfo("Sending waypoint %d - %s", i, w[0])
		   rospy.init_node('')
       	           if eval(data[0])<4:
			data=eval(data)
		   	edited_goal=(data[0],data[1],data[2])

		   	p.set_goal_to_point(edited_goal)
		   	print(edited_goal)
		   elif eval(data[0])<6:
			sound_play_func("mask reul saw")
       	        	p.set_goal_to_point(w[1])
		   else:
			sound_play_func("thank you")
       	        	p.set_goal_to_point(w[1])
		
    except rospy.ROSInterruptException:
        rospy.logerr("Something went wrong when sending the waypoints")
	

