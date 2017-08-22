#!/usr/bin/env python

import sys, time
import rospy
import math
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import numpy as np
import cv2

class Distance:
    def __init__(self):
        self.depth_image = None
        self.bridge = CvBridge()
        self.center = 0
        rospy.init_node('distance', anonymous=True)
        rospy.Subscriber('/kinect2/qhd/image_depth_rect', Image, self.callback)

    def callback(self, data):
        image = self.bridge.imgmsg_to_cv2(data, "16UC1")
        self.depth_image =  np.copy(image)
        unique, counts = np.unique(self.depth_image, return_counts=True)
        h, w = self.depth_image.shape
        self.center =  (w/2,h/2)
        print self.depth_image[self.center[1], self.center[0]]

    def listen(self):
        while not rospy.is_shutdown():
            # # frame = cv2.resize(self.color_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            frame = np.copy(self.depth_image * 255)
            cv2.circle(frame, self.center, 4, 100000, -1)

            cv2.imshow("Frame", frame)

            key = cv2.waitKey(delay=1)
            if key == ord('q'):
                break


if __name__ == '__main__':
    d = Distance()
    time.sleep(5)
    d.listen()
    cv2.destroyAllWindows()
