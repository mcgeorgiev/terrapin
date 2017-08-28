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
import names
import matplotlib.pyplot as plt
from Marker import Mark_Maker
from google_query import GoogleVision
from tensor_flow.tf_files.label_image import TensorFlow

class Box:
    def __init__(self, x, y, w, h, scale):
        self.x = x * int(1/scale)
        self.y = y * int(1/scale)
        self.w = w * int(1/scale)
        self.h = h * int(1/scale)
        self.center_x = self.center_y = None
        self.get_center()
        self.roi = None

    def get_center(self):
        self.center_x = (self.x + self.x + self.w)/2
        self.center_y = (self.y + self.y + self.h)/2

    def crop(self, frame):
        self.roi = frame[self.x:self.x+self.w, self.y:self.y+self.h]


class Camera:
    def __init__(self, sensor):
        ###
        self.bridge = CvBridge()
        self.color_image = None
        self.depth_image = None
        self.gray_image = None
        self.sensor = sensor
        rospy.init_node('stream', anonymous=True)
        if self.sensor == "hack_kinect2":
            color_topic = "ir"
            depth_topic = "small_depth"
        elif self.sensor == "kinect2":
            color_topic = "/kinect2/qhd/image_color_rect"
            depth_topic = "/kinect2/qhd/image_depth_rect"
        elif self.sensor == "gazebo":
            color_topic = "/camera/rgb/image_raw"
            depth_topic = "/camera/depth/image_raw"

        rospy.Subscriber(color_topic, Image, self.callback)
        rospy.Subscriber(depth_topic, Image, self.depth_callback)


        self.depth_mask = None
        self.current_depth_threshold = 0
        self.dbscan_scale = 0.15
        # 1500
        self.depth_max_threshold = 1200
        self.mask = None
        self.labels = None
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.marker_object = Mark_Maker("gazebo")
        self.frame = None
        self.google_vision = GoogleVision()
        self.tensorflow = TensorFlow()


    def callback(self, data):
        if self.sensor == "hack_kinect2":
            image = self.bridge.imgmsg_to_cv2(data, "16UC1")
        elif self.sensor == "kinect2" or self.sensor == "gazebo":
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")

        self.color_image = np.copy(image)



    def depth_callback(self, data):
        if self.sensor == "hack_kinect2":
            image = self.bridge.imgmsg_to_cv2(data, "8UC1")
        elif self.sensor == "kinect2" or self.sensor == "gazebo":
            image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        depth_image = np.copy(image)
        if self.sensor == "gazebo":
            where_are_NaNs = np.isnan(depth_image)
            depth_image[where_are_NaNs] = 0
        # # depth_image = depth_image[:depth_image.shape[0]-2] # shave off
        depth_threshold = threshold_otsu(depth_image)
        if depth_threshold > self.depth_max_threshold:
            depth_threshold = self.depth_max_threshold
        #print depth_threshold
        depth_image[depth_image > depth_threshold] = 0
        depth_image[depth_image > 0] = 255
        # self.depth_mask = np.squeeze(depth_image < depth_threshold)
        self.depth_image = depth_image
        unique, counts = np.unique(self.depth_image, return_counts=True)
        # print unique
        # self.current_depth_thresh = depth_threshold

    def new_name(self):
        return str(names.get_full_name())

    # def show_boxes(self):
    #     # gets all points from the cluster
    #     all_points = {}
    #     unique, counts = np.unique(self.labels, return_counts=True)
    #     for label in unique:
    #         x, y =  np.where(self.reshaped_img == label)
    #         points =  np.column_stack((x, y))
    #         all_points[label] = points
    #         # x,y,w,h = cv2.boundingRect(points)
    #         # cv2.rectangle(labimg,(y,x),(y+h, x+w),100,1)
    #
    #     bounding_boxes = {}
    #     for key, value in all_points.items():
    #         x,y,w,h = cv2.boundingRect(value)
    #         x *= int(1/self.dbscan_scale)
    #         y *= int(1/self.dbscan_scale)
    #         w *= int(1/self.dbscan_scale)
    #         h *= int(1/self.dbscan_scale)
    #         # bounding_boxes[key] = Box(x, y, w, h)
    #         # bounding_boxes[key].update(self.color_image)
    #         cv2.putText(self.color_image, str(key),(y,x+50), self.font, 1,(0,0,0),3 ,cv2.LINE_AA)
    #         cv2.rectangle(self.color_image,(y,x),(y+h, x+w),(0,0,0),1)
    #     # reform the size of the bounding box
    #     bounding_boxes = {}
    #     for key, value in all_points.items():
    #         x,y,w,h = cv2.boundingRect(value)
    #         bounding_boxes[key] = Box(x, y, w, h, self.dbscan_scale)
    #
    #         # show the box
    #         cv2.putText(self.color_image, str(key),(bounding_boxes[key].y,bounding_boxes[key].x+50), self.font, 1,(0,0,0),3 ,cv2.LINE_AA)
    #         cv2.rectangle(self.color_image,(bounding_boxes[key].y,bounding_boxes[key].x),(bounding_boxes[key].y+bounding_boxes[key].h, bounding_boxes[key].x+bounding_boxes[key].w),(0,0,0),1)



    def box(self):
        all_points = {}
        unique, counts = np.unique(self.labels, return_counts=True)
        for label in unique:
            x, y =  np.where(self.reshaped_img == label)
            points =  np.column_stack((x, y))
            all_points[label] = points

        # reform the size of the bounding box
        bounding_boxes = {}
        for key, value in all_points.items():
            x,y,w,h = cv2.boundingRect(value)
            if len(value) > 1500:
                bounding_boxes[key] = Box(x, y, w, h, self.dbscan_scale)
                bounding_boxes[key].crop(self.color_image)

                # show the box
                existing_marker = self.marker_object.add_marker(bounding_boxes[key].center_x,bounding_boxes[key].center_y)
                if not existing_marker:
                    label, score = self.tensorflow.query(bounding_boxes[key].roi)
                    time = 0
                    if score < 0.5:
                        label, score = self.google_vision.query(bounding_boxes[key].roi)
                        time = 15
                        if score < 0.8: # if the google vision api gives a bad reading
                            continue
                    # cv2.imwrite("crop.png", bounding_boxes[key].roi)
                    self.marker_object.markerArray.markers[-1].ns = label
                    self.marker_object.markerArray.markers[-1].text = label
                else:
                    label = existing_marker.ns

                # show the box
                cv2.putText(self.frame, str(label),(bounding_boxes[key].y,bounding_boxes[key].x+50), self.font, 1,(0,255,0),3 ,cv2.LINE_AA)
                cv2.rectangle(self.frame,(bounding_boxes[key].y,bounding_boxes[key].x),(bounding_boxes[key].y+bounding_boxes[key].h, bounding_boxes[key].x+bounding_boxes[key].w),(0,255,0),4)



    def segment(self):
        try:
            if self.depth_mask.all() == None:
                return
        except AttributeError as e:
            pass

        if self.sensor == "hack_kinect2":
            self.gray_image = self.color_image
        elif self.sensor == "kinect2" or self.sensor == "gazebo":
            self.gray_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)


        mask1 = self.gray_image < threshold_otsu(self.gray_image)
        mask2 = self.gray_image > threshold_otsu(self.gray_image)
        mask = mask1 if np.sum(mask1) < np.sum(mask2) else mask2
        # mask = morphology.remove_small_objects(mask, 50)
        # if self.sensor == "kinect2":
        #     # using this line always screws things up
        #     mask = morphology.remove_small_objects(mask * self.depth_mask, 50)
        mask = mask * self.depth_image
        mask = np.asarray(mask, dtype=np.uint8)
        mask = morphology.remove_small_objects(mask, 50)

        mask[mask > 0] = 255
        self.mask = mask
        # self.mask[self.mask > 0] = 255

        labimg = cv2.resize(self.depth_image, None, fx=self.dbscan_scale, fy=self.dbscan_scale, interpolation=cv2.INTER_NEAREST)
        labimg = cv2.cvtColor(labimg, cv2.COLOR_GRAY2BGR)
        rows, cols, channels = labimg.shape

        indices = np.dstack(np.indices(labimg.shape[:2]))
        xycolors = np.concatenate((labimg, indices), axis=-1)
        feature_image = np.reshape(xycolors, [-1,5])

        start =  time.time()
        # 10 and 50
        db = DBSCAN(eps=5, min_samples=30, metric = 'euclidean',algorithm ='auto')
        db.fit(feature_image)
        rospy.loginfo("Time for DBSCAN:" + str(time.time() - start))

        self.labels = db.labels_
        self.reshaped_img = np.reshape(self.labels, [rows, cols])
        unique, counts = np.unique(self.reshaped_img, return_counts=True)
        # for color in unique:
        # self.reshaped_img[self.reshaped_img == -1] = (255,0,0)
        # print unique

    def listen(self):
        while not rospy.is_shutdown():
            # self.show_boxes()
            #
            # ok, bbox = self.tracker.update(self.color_image)
            # time.sleep(0.25)
            # self.segment()

            # create frame so it can be manipulated
            self.frame = self.color_image
            # self.marker_object.add_marker(320, 240)
            # # frame = cv2.resize(self.color_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            # self.segment()
            # self.box()
            # x = 200
            # y = 200
            print "markers------------------"
            for i in range(0, len(self.marker_object.markerArray.markers)):
                print i, self.marker_object.markerArray.markers[i].ns

            cv2.imshow("Frame", self.frame)
            # plt.figure(4)
            # plt.subplot(4, 1, 1)
            # plt.imshow(self.mask)
            # plt.axis('off')
            # plt.subplot(4, 1, 2)
            # plt.imshow(self.depth_image)
            # plt.axis('off')
            # plt.subplot(4, 1, 3)
            # plt.imshow(self.reshaped_img)
            # plt.axis('off')
            # plt.subplot(4, 1, 4)
            # plt.imshow(self.color_image)
            # plt.axis('off')
            # plt.show()
            key = cv2.waitKey(delay=1)
            if key == ord('q'):
                break


if __name__ == '__main__':
    c = Camera('gazebo')
    time.sleep(5)
    c.listen()
    cv2.destroyAllWindows()
