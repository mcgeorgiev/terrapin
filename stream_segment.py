import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import cv2
import numpy as np
import time
from skimage import morphology
from skimage.filters import threshold_otsu


depth_threshold = 255
cap = cv2.VideoCapture('cut_tiger.avi')



while(cap.isOpened()):
    # time.sleep(.3)
    frame_start = time.time()
    ret, frame = cap.read()
    gray_image = frame
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mask1 = gray_image < threshold_otsu(gray_image)
    mask2 = gray_image > threshold_otsu(gray_image)
    mask = mask1 if np.sum(mask1) < np.sum(mask2) else mask2

    mask = morphology.remove_small_objects(mask * depth_threshold, 50)
    mask = np.asarray(mask)
    # make 3d
    mask = mask.reshape(mask.shape[0],mask.shape[1],1).astype(np.uint8)
    # print mask.dtype
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    labimg = cv2.cvtColor(mask, cv2.COLOR_BGR2LAB)
    labimg = cv2.resize(mask, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_NEAREST)
    rows, cols, channels = labimg.shape

    indices = np.dstack(np.indices(labimg.shape[:2]))

    xycolors = np.concatenate((labimg, indices), axis=-1)
    feature_image = np.reshape(xycolors, [-1,5])

    # 17 for 0.5
    # 10 for 0.25
    start =  time.time()
    db = DBSCAN(eps=10, min_samples=50, metric = 'euclidean',algorithm ='auto')
    db.fit(feature_image)
    print time.time() - start
    labels = db.labels_

    unique, counts = np.unique(labels, return_counts=True)
    reshaped_img = np.reshape(labels, [rows, cols])

    # gets all points from the cluster
    all_points = {}
    for label in unique:
        x, y =  np.where(reshaped_img == label)
        points =  np.column_stack((x, y))
        all_points[label] = points
        x,y,w,h = cv2.boundingRect(points)
        cv2.rectangle(labimg,(y,x),(y+h, x+w),100,1)

    # reform the size of the bounding box
    for key, value in all_points.items():
        x,y,w,h = cv2.boundingRect(value)
        x *= 5
        y *= 5
        w *= 5
        h *= 5
        cv2.rectangle(frame,(y,x),(y+h, x+w),(255,255,255),1)

    print "frame length", time.time() - frame_start

    ### Display the frame
    if DEBUG:
        plt.figure(2)
        plt.subplot(2, 1, 1)
        plt.imshow(frame)
        plt.axis('off')
        plt.subplot(2, 1, 2)
        plt.imshow(reshaped_img)
        plt.axis('off')
        plt.show()
    else:
        cv2.imshow('frame',frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
