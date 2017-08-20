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
from Box import Box
import names

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
        self.labels = None

        self.current_bounding_boxes = {}
        self.font = cv2.FONT_HERSHEY_SIMPLEX


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

        self.labels = db.labels_
        self.reshaped_img = np.reshape(self.labels, [rows, cols])

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

    def new_name(self):
        return str(names.get_full_name())

    def show_boxes(self):
        # gets all points from the cluster
        all_points = {}
        unique, counts = np.unique(self.labels, return_counts=True)
        for label in unique:
            x, y =  np.where(self.reshaped_img == label)
            points =  np.column_stack((x, y))
            all_points[label] = points
            # x,y,w,h = cv2.boundingRect(points)
            # cv2.rectangle(labimg,(y,x),(y+h, x+w),100,1)

        # reform the size of the bounding box
        bounding_boxes = {}
        for key, value in all_points.items():
            x,y,w,h = cv2.boundingRect(value)
            x *= int(1/self.dbscan_scale)
            y *= int(1/self.dbscan_scale)
            w *= int(1/self.dbscan_scale)
            h *= int(1/self.dbscan_scale)
            bounding_boxes[key] = Box(x, y, w, h)
            bounding_boxes[key].update(self.color_image)
            cv2.rectangle(self.color_image,(y,x),(y+h, x+w),(0,0,0),1)


        # removes the whole frame from the bounding boxes

        try:
            del bounding_boxes[0]
            del bounding_boxes[-1]
        except:
            pass

        # If the current boxes are empty (first frame has started)
        if not self.current_bounding_boxes:
            temp_boxes = bounding_boxes.copy()
            for key in temp_boxes.keys():
                name = self.new_name()
                self.current_bounding_boxes[name] = temp_boxes.pop(key)
            rospy.loginfo("Got new bounding boxes: \n" + str(self.current_bounding_boxes))

        # else:
        #     delete_keys = []
        #     temp_current = {}
        #     for cur_name in self.current_bounding_boxes.keys():
        #         for bb_name in bounding_boxes.keys():
        #
        #             if self.current_bounding_boxes[cur_name].is_match(bounding_boxes[bb_name]):
        #                 print cur_name, "matches"
        #                 temp_current[cur_name] = bounding_boxes[bb_name]
        #                 break
        #     self.current_bounding_boxes = temp_current.copy()
        #
        #     print len(self.current_bounding_boxes), self.current_bounding_boxes.keys()
        #     print len(bounding_boxes), bounding_boxes.keys()

        # # try:
        # print current_bounding_boxes
        # print "names"
        # for key in current_bounding_boxes.keys():
        #     print key
        #     cv2.putText(frame, key,(current_bounding_boxes[key].x,current_bounding_boxes[key].y), font, 1,(0,0,0),3 ,cv2.LINE_AA)



    def listen(self):

        while not rospy.is_shutdown():
            self.show_boxes()
            # frame = cv2.resize(self.color_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            cv2.imshow("depth", self.color_image)

            key = cv2.waitKey(delay=1)
            if key == ord('q'):
                break


if __name__ == '__main__':
    c = Camera()
    time.sleep(5)
    c.listen()
    cv2.destroyAllWindows()
