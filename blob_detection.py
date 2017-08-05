import cv2
import numpy as np
import math
import time

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

        self.id = 0

    def add(self, _x, _y):
        self.minx = min(self.minx, _x)
        self.miny = min(self.miny, _y)
        self.maxx = max(self.maxx, _x)
        self.maxy = max(self.maxy, _y)

        self.centerx = (self.minx + self.maxx) / 2
        self.centery = (self.miny + self.maxy) / 2

        self.points.append((_x, _y))

    def euclidean_distance(self, _x, _y, a, b):
        return math.sqrt(((_x-a)**2)+((_y-b)**2))

    def is_near(self, _x, _y):
        distance = self.euclidean_distance(_x, _y, self.centerx, self.centery)
        return True if (distance < 190) else False

    def is_near_edge(self, _x, _y):
        shortest_distance = 100000
        for vector in self.points:
            temp_distance = self.euclidean_distance(_x, _y, vector[0], vector[1])
            if temp_distance < shortest_distance:
                shortest_distance = temp_distance

        return True if (shortest_distance < 50) else False

    def is_near_clamp(self, _x, _y):
        x = max(min(_x, self.maxx), self.minx)
        y = max(min(_y, self.maxy), self.miny)
        distance = self.euclidean_distance(_x, _y, x, y)
        return True if (distance < 49) else False


blobs = []

def fast(filename):
    start_time = time.time()

    current_blobs = []
    img = cv2.imread(filename,0)

    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector_create()

    # find and draw the keypoints
    # print type(img)
    # print type(fast)
    keypoints = fast.detect(img,None)
    # print [point.pt for point in kp]
    img2 = cv2.drawKeypoints(img, keypoints, img, color=(255,0,0))

    # Print all default params
    # print "Total Keypoints with nonmaxSuppression: ", len(keypoints)

    kp_coords = [point.pt for point in keypoints]
    # print kp_coords

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
        if len(blob.points) > 100:
            temp_blobs.append(blob)
            cv2.rectangle(img2, (blob.minx, blob.miny), (blob.maxx, blob.maxy), (0,255,0), 1)
            x = blob.minx
            y = blob.miny
            w = blob.maxx-blob.minx
            h = blob.maxy-blob.miny
            print x, y, h, w

            roi = img[y:y+h, x:x+w]
            cv2.imwrite(str(blob)+'_roi.png',roi)

    print("--- %s seconds ---" % (time.time() - start_time))
    current_blobs = temp_blobs
    print len(current_blobs)
    # match current blobs to blobs

    cv2.imwrite('fast_true.png',img2)





if "__main__" == __name__:
    filename = 'objects.jpg'
    fast(filename)
