import numpy as np
import cv2

img1 = cv2.imread('coke.jpg',0)          # queryImage
img2 = cv2.imread('drinking_coke.jpg',0) # trainImage

# Initiate SIFT fast
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
img3 = None
# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], img3, flags=2)

cv2.imwrite('match.png',img3)
