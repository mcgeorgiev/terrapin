#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import cv2
import time
from cv_bridge import CvBridge
import numpy as np
from skimage.filters import threshold_otsu
from skimage import morphology
from sklearn.cluster import DBSCAN
import pylab as plt


class Camera:
    def __init__(self):
        ###
        self.bridge = CvBridge()
        self.color_image = None
        self.depth_image = None
        self.gray_image = None
        rospy.init_node('stream', anonymous=True)
        rospy.Subscriber("ir", Image, self.callback)
        rospy.Subscriber("small_depth", Image, self.depth_callback)

        self.depth_mask = None
        self.current_depth_threshold = 0
        self.dbscan_scale = 0.25
        self.depth_max_threshold = 1000
        self.mask = None

    def callback(self, data):
        image = self.bridge.imgmsg_to_cv2(data, "16UC1")
        self.color_image = np.copy(image)

        if self.depth_mask.all() == None:
            return
        self.gray_image = self.color_image #cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)

        mask1 = self.gray_image < threshold_otsu(self.gray_image)
        mask2 = self.gray_image > threshold_otsu(self.gray_image)
        mask = mask1 if np.sum(mask1) < np.sum(mask2) else mask2


        # using this line always screws things up
        # mask = morphology.remove_small_objects(mask * self.depth_mask, 50)
        mask = morphology.remove_small_objects(mask, 50)
        mask = np.asarray(mask, dtype=np.uint8)
        self.mask = mask
        self.mask[self.mask > 0] = 255

        labimg = cv2.resize(self.mask, None, fx=self.dbscan_scale, fy=self.dbscan_scale, interpolation=cv2.INTER_NEAREST)
        labimg = cv2.cvtColor(labimg, cv2.COLOR_GRAY2BGR)
        rows, cols, channels = labimg.shape

        indices = np.dstack(np.indices(labimg.shape[:2]))
        xycolors = np.concatenate((labimg, indices), axis=-1)
        feature_image = np.reshape(xycolors, [-1,5])

        start =  time.time()
        db = DBSCAN(eps=10, min_samples=50, metric = 'euclidean',algorithm ='auto')
        db.fit(feature_image)
        rospy.loginfo("Time for DBSCAN:" + str(time.time() - start))

        labels = db.labels_
        self.reshaped_img = np.reshape(labels, [rows, cols])

        # plt.figure(4)
        # plt.subplot(4, 1, 1)
        # plt.imshow(self.mask)
        # plt.axis('off')
        # plt.subplot(4, 1, 2)
        # plt.imshow(self.depth_mask)
        # plt.axis('off')
        # plt.subplot(4, 1, 3)
        # plt.imshow(self.reshaped_img)
        # plt.axis('off')
        # plt.subplot(4, 1, 4)
        # plt.imshow(self.color_image)
        # plt.axis('off')
        # plt.show()

    def depth_callback(self, data):
        image = self.bridge.imgmsg_to_cv2(data, "8UC1")
        depth_image = np.copy(image)
        # depth_image = depth_image[:depth_image.shape[0]-2] # shave off

        depth_threshold = threshold_otsu(depth_image)

        if depth_threshold > self.depth_max_threshold:
            depth_threshold = self.depth_max_threshold

        depth_image[depth_image == 0] = depth_threshold
        self.depth_mask = np.squeeze(depth_image < depth_threshold)
        self.depth_image = depth_image
        self.current_depth_thresh = depth_threshold

    def listen(self):

        while not rospy.is_shutdown():
            # self.mask[self.mask > 0] = 255
            # unique, counts = np.unique(self.mask, return_counts=True)
            # rospy.loginfo(unique)
            # rospy.loginfo(counts)
            # frame = cv2.resize(self.reshaped_img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

            # cv2.imshow("depth", frame)


            key = cv2.waitKey(delay=1)
            if key == ord('q'):
                break


if __name__ == '__main__':
    c = Camera()
    time.sleep(5)
    c.listen()
    cv2.destroyAllWindows()
