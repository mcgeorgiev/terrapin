#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import cv2
import time
from cv_bridge import CvBridge
import numpy as np
from numpy import inf
from skimage.filters import threshold_otsu
from skimage import morphology
from sklearn.cluster import DBSCAN
import names
import matplotlib.pyplot as plt
from Marker import Mark_Maker
from google_query import GoogleVision
from tensor_flow.tf_files.label_image import TensorFlow
from sklearn.preprocessing import StandardScaler
from sklearn import mixture

class Box:
    def __init__(self, x, y, w, h, scale):
        self.x = x * int(1/scale)
        self.y = y * int(1/scale)
        self.w = w * int(1/scale)
        self.h = h * int(1/scale)
        self.center_x = self.center_y = None
        self.get_center()
        self.roi = None
        self.corners = 0

    def get_center(self):
        self.center_x = (self.x + self.x + self.w)/2
        self.center_y = (self.y + self.y + self.h)/2

    def crop(self, frame):
        self.roi = frame[self.x:self.x+self.w, self.y:self.y+self.h]
        self.fast()

    def to_big(self, full_image):
        if self.h > (full_image.shape[0]/1.5) or self.w > (full_image.shape[1]/1.5):
            return True

    def fast(self):
        # Initiate FAST object with default values
        fast = cv2.FastFeatureDetector_create()
        kp = fast.detect(self.roi,None)
        self.corners = len(kp)

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
        elif self.sensor == "zed":
            color_topic = "/rgb/image_rect_color"
            depth_topic = "/depth/depth_registered"

        rospy.Subscriber(color_topic, Image, self.callback)
        rospy.Subscriber(depth_topic, Image, self.depth_callback)


        self.depth_mask = None
        self.current_depth_threshold = 0
        self.dbscan_scale = 0.12
        # 1500
        self.depth_max_threshold = 2000
        self.mask = None
        self.labels = None
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.marker_object = Mark_Maker(self.sensor)
        self.frame = None
        self.google_vision = GoogleVision()
        self.tensorflow = TensorFlow()

        self.gmm = mixture.GMM(n_components=1, covariance_type='full')
        self.average_color = self.set_floor_hsv()


    def callback(self, data):
        if self.sensor == "hack_kinect2":
            image = self.bridge.imgmsg_to_cv2(data, "16UC1")
        elif self.sensor == "kinect2" or self.sensor == "gazebo" or self.sensor == "zed":
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")

        self.color_image = np.copy(image)



    def depth_callback(self, data):
        if self.sensor == "hack_kinect2":
            image = self.bridge.imgmsg_to_cv2(data, "8UC1")
        elif self.sensor == "kinect2" or self.sensor == "gazebo" or self.sensor == "zed":
            image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        depth_image = np.copy(image)
        if self.sensor == "gazebo" or self.sensor == "zed":
            where_are_NaNs = np.isnan(depth_image)
            depth_image[where_are_NaNs] = 0
        if self.sensor == "zed":
            depth_image *= 1000 #zed cam is in meters while processing happens in mm
            depth_image[depth_image == -inf] = 0
            depth_image[depth_image >= 1E308] = 0
        # # # depth_image = depth_image[:depth_image.shape[0]-2] # shave off
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
        self.current_depth_thresh = depth_threshold

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

                if bounding_boxes[key].to_big(self.color_image):
                    print key, "too big"
                    continue
                # if bounding_boxes[key].corners < 500:
                #     continue

                if key in [-1]:
                    continue

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

    def rand_color(self):
        return int(np.random.random() * 255)

    def segment(self):
        try:
            if self.depth_mask.all() == None:
                return
        except AttributeError as e:
            pass

        if self.sensor == "hack_kinect2":
            self.gray_image = self.color_image
        elif self.sensor == "kinect2" or self.sensor == "gazebo" or self.sensor == "zed":
            self.gray_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)

        self.hsv = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2HSV) # assuming converted is your original stream...

        # skinRegion = cv2.inRange(hsv,min_YCrCb,max_YCrCb) # Create a mask with boundaries
        # contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find the contour on the skin detection
        #     for i, c in enumerate(contours): # Draw the contour on the source frame
        #         area = cv2.contourArea(c)
        #         if area > 10000:
        #             cv2.drawContours(img, contours, i, (255, 255, 0), 2)


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

        blurred_hsv = cv2.medianBlur(self.hsv,15)
        hsv_mask = cv2.inRange(blurred_hsv, self.average_color*0.1, self.average_color*1.9)
        self.mask = self.mask - hsv_mask




        # X, x_size, y_size = self.create_features(self.mask, self.gray_image)
        #
        # boxes = []
        # if X.shape[0] > 100:
        #     X_prime = np.copy(X)
        #     X_prime = StandardScaler().fit_transform(X_prime)
        #     db = DBSCAN(eps=0.7, min_samples=20).fit(X_prime)
        #     n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        #     labels = db.labels_
        #     unique_labels = set(labels)
        #     cluster_image = np.asarray(self.mask, dtype=np.uint8) * 255
        #     self.cluster_image = np.repeat(cluster_image, 3, axis=1).reshape(self.mask.shape + (3, ))
        #
        #     for k in unique_labels:
        #
        #         # this is the sklean label for noise
        #         if k == -1:
        #             continue
        #
        #         class_member_mask = (labels == k)
        #
        #         class_feats = X[class_member_mask, :]
        #
        #         # probably a noisey cluster detection
        #         if class_feats.shape[0] < 20:
        #             continue
        #
        #
        #         self.gmm.fit(class_feats)
        #         covars = np.sqrt(np.asarray(self.gmm._get_covars()))
        #         alpha = 2.2
        #         x1 = int((self.gmm.means_[0, 0] - alpha * covars[0, 0, 0]) * x_size)
        #         x2 = int((self.gmm.means_[0, 0] + alpha * covars[0, 0, 0]) * x_size)
        #         y1 = int((self.gmm.means_[0, 1] - alpha * covars[0, 1, 1]) * y_size)
        #         y2 = int((self.gmm.means_[0, 1] + alpha * covars[0, 1, 1]) * y_size)
        #         # d1 = int((self.gmm.means_[0, 3] - alpha * covars[0, 3, 3]) * self.current_depth_thresh)
        #         # d2 = int((self.gmm.means_[0, 3] + alpha * covars[0, 3, 3]) * self.current_depth_thresh)
        #
        #         mean_depth = self.gmm.means_[0, 3] * self.current_depth_thresh
        #         boxes.append([x1, y1, x2, y2, mean_depth])
        #
        #         exp=5
        #         rc = self.rand_color()
        #         rc2 = self.rand_color()
        #         rc3 = self.rand_color()
        #         # c = self.gmm.means_[0, 3] * 255
        #         for i in range(class_feats.shape[0]):
        #             x, y = int(class_feats[i,0]*x_size), int(class_feats[i,1]*y_size)
        #             self.cluster_image[max(x- exp,0):min(x+exp, self.cluster_image.shape[0]),max(y- exp,0):min(y+exp,self.cluster_image.shape[1]), 0] = rc3
        #             self.cluster_image[max(x- exp,0):min(x+exp, self.cluster_image.shape[0]),max(y- exp,0):min(y+exp,self.cluster_image.shape[1]), 1] = rc2
        #             self.cluster_image[max(x- exp,0):min(x+exp, self.cluster_image.shape[0]),max(y- exp,0):min(y+exp,self.cluster_image.shape[1]), 2] = rc
        #
        #
        #     # if self.show_segmentation:
        #     for x1, y1, x2, y2, mean_depth in boxes:
        #         # cv2.putText(self.frame, str(k),(y1,x1+50), self.font, 1,(0,0,0),3 ,cv2.LINE_AA)
        #         cv2.rectangle(self.frame, (y1, x1), (y2, x2), (125,0,0), 2)



        labimg = cv2.resize(self.mask, None, fx=self.dbscan_scale, fy=self.dbscan_scale, interpolation=cv2.INTER_NEAREST)
        labimg = cv2.cvtColor(labimg, cv2.COLOR_GRAY2BGR)
        rows, cols, channels = labimg.shape

        indices = np.dstack(np.indices(labimg.shape[:2]))
        xycolors = np.concatenate((labimg, indices), axis=-1)
        feature_image = np.reshape(xycolors, [-1,5])

        start =  time.time()
        # 10 and 50
        # 5 and 30
        db = DBSCAN(eps=5, min_samples=30, metric = 'euclidean',algorithm ='auto')
        db.fit(feature_image)
        rospy.loginfo("Time for DBSCAN:" + str(time.time() - start))

        self.labels = db.labels_
        self.reshaped_img = np.reshape(self.labels, [rows, cols])
        # unique, counts = np.unique(self.reshaped_img, return_counts=True)
        # for color in unique:
        # self.reshaped_img[self.reshaped_img == -1] = (255,0,0)
        # print unique




    def create_features(self, mask, gray_image):
        x_size, y_size = mask.shape[0], mask.shape[1]
        gX, gY = np.meshgrid(range(mask.shape[0]), range(mask.shape[1]))
        X = np.zeros((mask.size, 5))
        X[:, 0] = gX.transpose().ravel() * 1.0 / x_size
        X[:, 1] = gY.transpose().ravel() * 1.0 / y_size
        X[:, 2] = mask.ravel()
        X[:, 3] = self.depth_image.ravel() * 1.0 / self.current_depth_thresh
        X[:, 4] = gray_image.ravel() * 1.0 / gray_image.max()
        # X = X[X[:, 2] == 1]
        num_samples = X.shape[0]
        step_size = int(num_samples / 3000)

        if step_size == 0:
            step_size = 1

        X = X[0::step_size, :]
        return X, x_size, y_size

    def set_floor_hsv(self):
        # region = self.hsv[600:800,700:1000]
        #
        # average_color_per_row = np.average(region, axis=0)
        # average_color = np.average(average_color_per_row, axis=0)
        # print average_color
        return np.asarray([  17.17480556,   51.27622222,  105.36063889])

    def listen(self):
        while not rospy.is_shutdown():
            # self.show_boxes()
            #
            # ok, bbox = self.tracker.update(self.color_image)
            # time.sleep(0.25)
            # self.frame = self.mask
            self.frame = self.color_image

            self.segment()
            # self.box()
            # create frame so it can be manipulated
            # self.frame = self.color_image
            # self.marker_object.add_marker(320, 240)
            # # frame = cv2.resize(self.color_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            self.box()
            # x = 200
            # y = 200
            # print "markers------------------"
            # for i in range(0, len(self.marker_object.markerArray.markers)):
            #     print i, self.marker_object.markerArray.markers[i].ns

            # cv2.imshow("Frame", self.frame)
            # cv2.circle(self.hsv,(800,600), 60, (0,0,100), -1)

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
    c = Camera('zed')
    time.sleep(5)
    c.listen()
    cv2.destroyAllWindows()
