from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import cv2
import numpy as np

img = cv2.imread("cluster.png",1)

labimg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# n = 0
# while(n<3):
#     labimg = cv2.pyrDown(labimg)
#     n = n+1
# print labimg.shape
labimg = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)

rows, cols, chs = labimg.shape

indices = np.dstack(np.indices(labimg.shape[:2]))

xycolors = np.concatenate((labimg, indices), axis=-1)
feature_image = np.reshape(xycolors, [-1,5])


# 17 for 0.5
db = DBSCAN(eps=10, min_samples=50, metric = 'euclidean',algorithm ='auto')
db.fit(feature_image)
labels = db.labels_

unique, counts = np.unique(labels, return_counts=True)
# counter = dict(zip(unique, counts))
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
colour_img = img #cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
for key, value in all_points.items():
    x,y,w,h = cv2.boundingRect(value)
    x *= 4
    y *= 4
    w *= 4
    h *= 4
    cv2.rectangle(colour_img,(y,x),(y+h, x+w),(255,255,255),1)
# cv2.imwrite("reduced.png", labimg)
cv2.imwrite("full.png", colour_img)

# plt.figure(2)
# plt.subplot(2, 1, 1)
# plt.imshow(img)
# plt.axis('off')
# plt.subplot(2, 1, 2)
# plt.imshow(reshaped_img)
# plt.axis('off')
# plt.show()
