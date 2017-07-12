#!/usr/bin/env python

import sys
import rospy
from tf import transformations
import math
from nav_msgs.msg import Odometry

# class Position:
#     def __init__(self):
#         rospy.init_node('show_orientation', anonymous=False)
#
#         pos_topic = "/nav_msgs/Odometry"
#         self.pos_sub = rospy.Subscriber(pos_topic, Odometry, self.yaw_callback)
#         print self.pos_sub
#         rospy.spin();
#
#     def yaw_callback(self, data):
#         print transformations.euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
#
#

def odometryCb(msg):
    # print msg.pose.pose.orientation
    print transformations.euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
    # in radians


if __name__ == '__main__':
    rospy.init_node('odometry', anonymous=True)
    rospy.Subscriber('odom', Odometry, odometryCb)
    rospy.spin()
    # try:
    #     Position()
    # except rospy.ROSInterruptException:
    #     rospy.loginfo("exception")
