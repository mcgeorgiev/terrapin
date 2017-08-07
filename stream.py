import numpy as np
import cv2

import rospy
from sensor_msgs.msg import Image

from cv_bridge import CvBridge, CvBridgeError
import time

class Camera:
    def __init__(self):
        rospy.init_node('streamer', anonymous=True)
        rospy.Subscriber("camera/rgb/image_raw", Image, self.callback)
        rospy.on_shutdown(self.shutdown)

        self.bridge = CvBridge()
        self.image = None

    def callback(self, image_data):
        try:
            self.image = self.bridge.imgmsg_to_cv2(image_data, "bgr8")
        except CvBridgeError as e:
            print e

    def stream(self):
        while not rospy.is_shutdown():
            if self.image != None:
                time.sleep(0.2)
                cv2.imshow("Camera window", self.fast())
                cv2.waitKey(3)


    def shutdown(self):
        cv2.destroyAllWindows()
        print "shutting down..."

    def fast(self):
        # Initiate FAST object with default values
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fast = cv2.FastFeatureDetector_create()

        # find and draw the keypoints)
        kp = fast.detect(self.image,None)
        # get all the kp coordinates
        pts = [kp[idx].pt for idx in range(len(kp))]
        return cv2.drawKeypoints(self.image, kp, self.image, color=(255,0,0))


if __name__ == "__main__":
    c = Camera()
    c.stream()
