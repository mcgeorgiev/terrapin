import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import cv2
import numpy as np
import time
from skimage import morphology
from skimage.filters import threshold_otsu
import names

DEBUG = False
MIN_MATCH_COUNT = 10
depth_threshold = 255
cap = cv2.VideoCapture('cut_tiger.avi')

def create_mask(gray):
    mask1 = gray_image < threshold_otsu(gray_image)
    mask2 = gray_image > threshold_otsu(gray_image)
    mask = mask1 if np.sum(mask1) < np.sum(mask2) else mask2

    mask = morphology.remove_small_objects(mask * depth_threshold, 50)
    mask = np.asarray(mask)
    # make 3d
    mask = mask.reshape(mask.shape[0],mask.shape[1],1).astype(np.uint8)
    # print mask.dtype
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask


def create_sift(previous_frame, frame):
    if previous_frame.size == 0:
        return

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(previous_frame,None)
    kp2, des2 = sift.detectAndCompute(frame,None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        print "Good match"
        # src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        # dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        #
        # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        # matchesMask = mask.ravel().tolist()


    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None


def debug_show(frame, reshaped_img):
    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.imshow(frame)
    plt.axis('off')
    plt.subplot(2, 1, 2)
    plt.imshow(reshaped_img)
    plt.axis('off')
    plt.show()

def new_name():
    return str(names.get_full_name())

class Box:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.crop = None

    def update(self, frame):
        self.crop = frame[self.x:self.x+self.w, self.y:self.y+self.h]

    def is_match(self, previous_crop):
        if previous_crop.crop.size == 0:
            return

        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(previous_crop.crop,None)
        kp2, des2 = sift.detectAndCompute(self.crop,None)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        if len(good)>MIN_MATCH_COUNT:
            print "Good match"
            return True
        else:
            print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
            return False

coke = cv2.imread("coke.jpg")

font = cv2.FONT_HERSHEY_SIMPLEX
previous_crops = []
current_bounding_boxes = {}
n = 0
while(cap.isOpened()):
    # time.sleep(.3)



    frame_start = time.time()
    ret, frame = cap.read()

    # resize
    # frame = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
    cv2.circle(frame,(100,100), 64, (200,0,200), -1)
    n+=1
    if n>5 and n<10:
        frame[400:400+coke.shape[0], 50:50+coke.shape[1]] = coke




    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mask = create_mask(gray_image)

    labimg = cv2.cvtColor(mask, cv2.COLOR_BGR2LAB)
    labimg = cv2.resize(mask, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_NEAREST)
    rows, cols, channels = labimg.shape

    # create feature_image
    indices = np.dstack(np.indices(labimg.shape[:2]))
    xycolors = np.concatenate((labimg, indices), axis=-1)
    feature_image = np.reshape(xycolors, [-1,5])

    # perform dbscan
    # 17 for 0.5
    # 10 for 0.25
    start =  time.time()
    db = DBSCAN(eps=17, min_samples=50, metric = 'euclidean',algorithm ='auto')
    db.fit(feature_image)
    print time.time() - start

    # reshape using dbscan labels
    labels = db.labels_
    reshaped_img = np.reshape(labels, [rows, cols])

    # gets all points from the cluster
    all_points = {}
    unique, counts = np.unique(labels, return_counts=True)
    for label in unique:
        x, y =  np.where(reshaped_img == label)
        points =  np.column_stack((x, y))
        all_points[label] = points
        # x,y,w,h = cv2.boundingRect(points)
        # cv2.rectangle(labimg,(y,x),(y+h, x+w),100,1)

    # reform the size of the bounding box
    bounding_boxes = {}
    for key, value in all_points.items():
        x,y,w,h = cv2.boundingRect(value)
        x *= 10
        y *= 10
        w *= 10
        h *= 10
        bounding_boxes[key] = Box(x, y, w, h)
        bounding_boxes[key].update(frame)
        cv2.rectangle(frame,(y,x),(y+h, x+w),(255,255,255),1)


    # removes the whole frame from the bounding boxes
    try:
        del bounding_boxes[0]
    except:
        pass

    # If the current boxes are empty (first frame has started)
    print "******************* ", current_bounding_boxes
    if not current_bounding_boxes:
        print "inside"
        temp_boxes = bounding_boxes.copy()
        for key in temp_boxes.keys():
            current_bounding_boxes[new_name()] = temp_boxes.pop(key)
        print "Got new bounding boxes", current_bounding_boxes
    else:
        # if
        delete_keys = []
        temp_current = {}
        for cur_name in current_bounding_boxes.keys():
            for bb_name in bounding_boxes.keys():
                if current_bounding_boxes[cur_name].is_match(bounding_boxes[bb_name]):
                    print cur_name, "matches"
                    temp_current[cur_name] = bounding_boxes[bb_name]
                    break
        current_bounding_boxes = temp_current.copy()

        print len(current_bounding_boxes), current_bounding_boxes.keys()
        print len(bounding_boxes), bounding_boxes.keys()
    # try:
    print current_bounding_boxes
    print "names"
    for key in current_bounding_boxes.keys():
        print key
        cv2.putText(frame, key,(current_bounding_boxes[key].x,current_bounding_boxes[key].y), font, 1,(0,0,0),3 ,cv2.LINE_AA)
    # need to add the third case





    ### remember to only show bounding boxes of a certain size. BIGGER means longer, perhaps without homography
    # start = time.time()
    # for name in bounding_boxes:
    #     for previous_crop in previous_crops:
    #         bounding_boxes[name].is_match(previous_crop)


    print "Time for match: >>>>>>>>>>>>", time.time() - start



    print "frame length", time.time() - frame_start

    ### Display the frame
    if DEBUG:
        debug_show(frame, reshaped_img)

    else:
        cv2.imshow('frame',frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

    # reset and reform the crops
    previous_crops = []
    for name in bounding_boxes:
        previous_crops.append(bounding_boxes[name].crop)

cap.release()
cv2.destroyAllWindows()
