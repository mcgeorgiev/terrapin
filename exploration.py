#!/usr/bin/env python
import numpy as np
import cv2

import rospy
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry

import time

class Mapper:
    def __init__(self):
        self.resolution = None
        self.robot_x = None
        self.robot_y = None
        self.mapData = None
        self.map_grid = None
        self.robot_index = None

    def show(self):
        pass

    def grid(self):
        map_arr = np.asarray(self.mapData.data, np.int8)
        self.map_grid = map_arr.reshape(self.mapData.info.height, self.mapData.info.width)
        self.map_grid[self.map_grid == 0] = 50 # change color

    def map_callback(self, data):
        self.mapData = data
        self.grid()

    def odom_callback(self, data):
        x = data.pose.pose.position.x
        y = data.pose.pose.position.y
        try:
            xindex = int((x-self.mapData.info.origin.position.x) *(1.0/self.mapData.info.resolution))
            yindex = int((y-self.mapData.info.origin.position.y)*(1.0/self.mapData.info.resolution))
            self.robot_index = yindex * self.mapData.info.width + xindex
            # print "Point: ", self.robot_index
        except:
            print "not ready"

    def listen(self):
        rospy.init_node('map_listener', anonymous=True)
        rospy.Subscriber("map", OccupancyGrid, self.map_callback)
        rospy.Subscriber("odom", Odometry, self.odom_callback)
        time.sleep(1)
        # while True:
        self.robot_x = self.robot_index % self.mapData.info.width
        self.robot_y = self.robot_index / self.mapData.info.width
        cv2.circle(self.map_grid,(self.robot_x, self.robot_y), 3, (255,255,255), -1)
        cv2.imshow('image', self.map_grid)
        k = cv2.waitKey(0)
        if k == 27: # esc key
            cv2.destroyAllWindows()


if __name__ == '__main__':
    m = Mapper()
    m.listen()
