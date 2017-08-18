#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import cv2
import time
from cv_bridge import CvBridge
import numpy as np

class Tracker:
    def __init__(self):

        self.bridge = CvBridge()
        self.color_image = None
        rospy.init_node('stream', anonymous=True)
        rospy.Subscriber("color", Image, self.callback)

    def callback(self, data):
        image = self.bridge.imgmsg_to_cv2(data, "8UC4")
        self.color_image = np.copy(image)

    def listen(self):

        while not rospy.is_shutdown():
            frame = cv2.resize(self.color_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            cv2.imshow("color", frame)
            key = cv2.waitKey(delay=1)
            if key == ord('q'):
                break

if __name__ == '__main__':
    t = Tracker()
    time.sleep(5)
    t.listen()
    cv2.destroyAllWindows()
