import cv2
import numpy as np
import math
import time
import rospy
from sensor_msgs.msg import Image
import copy

from cv_bridge import CvBridge, CvBridgeError

def euclidean_distance(_x, _y, a, b):
    return math.sqrt(((_x-a)**2)+((_y-b)**2))

class Blob:
    def __init__(self, _x, _y):
        self.minx = _x
        self.miny = _y
        self.maxx = _x
        self.maxy = _y
        self.centerx = _x
        self.centery = _y
        self.points = []
        self.points.append((_x, _y))
        self.id = None

    def add(self, _x, _y):
        self.minx = min(self.minx, _x)
        self.miny = min(self.miny, _y)
        self.maxx = max(self.maxx, _x)
        self.maxy = max(self.maxy, _y)

        self.centerx = (self.minx + self.maxx) / 2
        self.centery = (self.miny + self.maxy) / 2

        self.points.append((_x, _y))

    def is_near(self, _x, _y):
        distance = euclidean_distance(_x, _y, self.centerx, self.centery)
        return True if (distance < 190) else False

    def is_near_edge(self, _x, _y):
        shortest_distance = 100000
        for vector in self.points:
            temp_distance = euclidean_distance(_x, _y, vector[0], vector[1])
            if temp_distance < shortest_distance:
                shortest_distance = temp_distance

        return True if (shortest_distance < 50) else False

    def is_near_clamp(self, _x, _y):
        x = max(min(_x, self.maxx), self.minx)
        y = max(min(_y, self.maxy), self.miny)
        distance = euclidean_distance(_x, _y, x, y)
        return True if (distance < 49) else False


class Camera:
    def __init__(self):
        rospy.init_node('streamer', anonymous=True)
        rospy.Subscriber("camera/rgb/image_raw", Image, self.callback)
        rospy.on_shutdown(self.shutdown)

        self.bridge = CvBridge()
        self.image = None
        self.blobs = []
        self.blob_counter = 0

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
        # start_time = time.time()
        current_blobs = []
        img = self.image

        # Initiate FAST object with default values
        fast = cv2.FastFeatureDetector_create()

        keypoints = fast.detect(img,None)
        img2 = cv2.drawKeypoints(img, keypoints, img, color=(255,0,0))

        kp_coords = [point.pt for point in keypoints]

        for kp in kp_coords:
            x = int(kp[0])
            y = int(kp[1])
            found = False
            for blob in current_blobs:
                if blob.is_near_edge(x, y):
                    blob.add(x, y)
                    found = True
                    break

            if not found:
                current_blobs.append(Blob(x, y))


        temp_blobs = []
        for blob in current_blobs:
            if len(blob.points) > 50:
                temp_blobs.append(blob)
                cv2.rectangle(img2, (blob.minx, blob.miny), (blob.maxx, blob.maxy), (0,255,0), 1)
        current_blobs = temp_blobs

        # print("--- %s seconds ---" % (time.time() - start_time))

        return img2


if __name__ == "__main__":
    c = Camera()
    c.stream()
