#!/usr/bin/env python

import sys, time
import rospy
import math
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2  as pc2
import numpy as np
import cv2
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

class Distance:
    def __init__(self):
        self.depth_image = None
        self.bridge = CvBridge()
        self.center = 0
        self.color = None
        self.point_3d_array = None

        self.markerArray = MarkerArray()
        self.count = 0
        self.MARKERS_MAX = 1

        rospy.init_node('distance', anonymous=True)

        rospy.Subscriber('/camera/depth/image_raw', Image, self.callback)
        # rospy.Subscriber('/depth/depth_registered', Image, self.callback)
        # rospy.Subscriber('camera/rgb/image_raw', Image, self.color_callback)
        # rospy.Subscriber('camera/depth/points', PointCloud2, self.point_cloud_callback)

        self.publisher = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)

    #     rospy.Subscriber('/kinect2/qhd/image_depth_rect', Image, self.callback)

    def callback(self, data):
        image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        self.depth_image =  np.copy(image)
        unique, counts = np.unique(self.depth_image, return_counts=True)
        # print unique,counts
        h, w = self.depth_image.shape
        self.center =  (w/2,h/2)
        # print self.depth_image[self.center[1], self.center[0]]
        # float raw_depth_to_meters(int raw_depth)
        # {
        #   if (raw_depth < 2047)
        #   {
        #    return 1.0 / (raw_depth * -0.0030711016 + 3.3309495161);
        #   }
        #   return 0;
        # }

    def color_callback(self, data):
        self.color = self.bridge.imgmsg_to_cv2(data, "bgr8")
        h, w, _ = self.color.shape
        self.center =  (w/2,h/2)


    def calculate_xyz(self):#, image, distance):
        # u = x position of point relative to origin (which is the center of the image)
        # v = y position of point
        # distance in meters from camera to point
        point = (520,120)
        distance_from_x_origin = abs(self.center[1] - point[1])
        # theta =

    def point_cloud_callback(self, data):
        point_cloud = data
        print "getting point data"
        point_list = []
        for p in pc2.read_points(point_cloud, field_names = ("x", "y", "z")):
            point_list.append((p[0],p[1],p[2]))
        point_array = np.array(point_list)
        self.point_3d_array = np.reshape(point_array, (480,640,3))
        # print self.point_3d_array[220][520]


    def get_xyz(self, x, y):
        return self.point_3d_array[y][x]

    def mark(self, x, y, z):
       marker = Marker()
       print marker
       marker.header.frame_id = "/base_link"
       marker.type = marker.SPHERE
       marker.action = marker.ADD
       marker.scale.x = 0.2
       marker.scale.y = 0.2
       marker.scale.z = 0.2
       marker.color.a = 1.0
       marker.color.r = 1.0
       marker.color.g = 1.0
       marker.color.b = 0.0
       marker.pose.orientation.w = 1.0
       marker.pose.position.x = x#(x+1)*0.5
       marker.pose.position.y = y#(y-1)*0.5
       marker.pose.position.z = z#(z*0.5)

       # We add the new marker to the MarkerArray, removing the oldest
       # marker from it when necessary
       if(self.count > self.MARKERS_MAX):
           self.markerArray.markers.pop(0)

       self.markerArray.markers.append(marker)

       # Renumber the marker IDs
       id = 0
       for m in self.markerArray.markers:
           m.id = id
           id += 1

       # Publish the MarkerArray
       self.publisher.publish(self.markerArray)

       self.count += 1


    def listen(self):
        while not rospy.is_shutdown():
            # frame = cv2.resize(self.color_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            # frame = np.copy(self.depth_image * 255)
            # cv2.circle(frame, self.center, 4, 100000, -1)
            frame = self.point_3d_array# np.copy(self.color)
            # cv2.circle(frame, self.center, 4, (255,255,255), -1)
            # cv2.circle(frame, (520,120), 4, (255,0,0), -1)
            # cv2.imshow("Frame", frame)
            # x, y, z = self.get_xyz(420,120)
            # print "Blue:",x,y,z
            # x, y, z = self.get_xyz(self.center[0],self.center[1])
            # print "White",x,y,z
            self.mark(1,1,1)
            key = cv2.waitKey(delay=1)
            if key == ord('q'):
                break









if __name__ == '__main__':
    d = Distance()
    time.sleep(5)
    d.listen()
    cv2.destroyAllWindows()
