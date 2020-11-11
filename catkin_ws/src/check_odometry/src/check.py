#! /usr/bin/env python
import sys
import rospy
from nav_msgs.msg import Odometry
#from sound_play.msg import SoundRequest
#import say
import socket
import tf
import time
sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
sock.connect(('172.22.77.105',7012))
#sock.connect(('172.22.77.105',8080))
#current_location={}
# = { 'X':0, 'Y':0, 'Z':0 }

def callback(msg):
	#global current_location 
	#current_location['X'] = msg.pose.pose.position.x
	#current_location['Y'] = msg.pose.pose.position.y
	#current_location['Z'] = msg.pose.pose.orientation.z
	#current_loc_lst=[current_location['X'],currnet_location['Y'],current_location['Z']]
	quaternion = (msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w)
	euler=tf.transformations.euler_from_quaternion(quaternion)
	
	sock.send(str([msg.pose.pose.position.x,msg.pose.pose.position.y,euler[2]]).encode())
	time.sleep(0.1)
	print str([msg.pose.pose.position.x,msg.pose.pose.position.y,euler[2]])
	#print str(current_location)
	#print current_location['X'],' ',current_location['Y'],' ',current_location['Z'],' \n'
	#print 'Y:',  msg.pose.pose.position.y
	#print 'Z:', msg.pose.pose.orientation.z
 	#print '\n' 

rospy.init_node('')
odom_sub=rospy.Subscriber('/odom', Odometry,callback)
#sock.send(current_location.encode())
rospy.spin()

