import cv2
import numpy as np
import os
import math

def process(filename, to_cut_image):
    print "---------------------->>> ", filename
    #Load the Image
    imgo = cv2.imread("raw_positives/"+ filename)
    # 60
    ret,thresh = cv2.threshold(imgo,60,255,cv2.THRESH_TRUNC)

    height, width = imgo.shape[:2]

    #Create a mask holder
    mask = np.zeros(imgo.shape[:2],np.uint8)

    #Grab Cut the object
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    #Hard Coding the Rect The object must lie within this rect.
    rect = (10,10,width-30,height-30)
    cv2.grabCut(thresh,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img1 = imgo*mask[:,:,np.newaxis]

    #Get the background
    background = imgo - img1

    #Change all pixels in the background that are not black to white
    background[np.where((background > [0,0,0]).all(axis = 2))] =[255,255,255]

    #Add the background and the image
    final = background + img1

    img_gray = cv2.cvtColor(final,cv2.COLOR_RGB2GRAY)
    height, width = final.shape[:2]
    rightmost = [0,0]

    # final[np.where((final != [255,255,255]).all(axis = 2))] = [0,255,255]

    points = np.argwhere(img_gray <150) # find where the black pixels are
    points = np.fliplr(points) # store them in x,y coordinates instead of row,col indices

    x, y, w, h = cv2.boundingRect(points) # create a rectangle around those points
    x, y, w, h = x-20, y-20, w+40, h+40


    center_x = (x+x+w)/2
    new_x = center_x-(h/2)
    crop = None
    # if h > w:
    #    # center_x = (x+x+w)/2
    cv2.line(final,(x,0),(x,1500),(255,255,0),5)
    cv2.line(final,(center_x,0),(center_x,1500),(255,0,0),5)
    cv2.line(final,(x+w,0),(x+w,1500),(255,0,255),5)
    # new_x = center_x-(h/2)
    crop = to_cut_image[y:y+h, new_x:new_x+h]
    # else:
    #     # center_x = (x+x+w)/2
    #     # new_x = center_x-(h/2)
    #     crop = to_cut_image[y:y+w, new_x:new_x+w]

    # cv2.rectangle(to_cut_image,(new_x, y),((new_x+h),y+h),(0,255,0),3)
    # crop = imgo[y:y+h, new_x:new_x+h]
    print crop.shape
    cv2.imwrite("positives/" + filename, crop)

def resize(filename):

    # load up an image
    img = cv2.imread("positives/"+filename)
    print "positives/"+filename
    print img.shape

    TARGET_PIXEL_AREA = 500000

    ratio = float(img.shape[1]) / float(img.shape[0])
    new_h = int(math.sqrt(TARGET_PIXEL_AREA / ratio) + 0.5)
    new_w = int((new_h * ratio) + 0.5)

    img2 = cv2.resize(img, (new_w,new_h))
    print img2.shape
    cv2.imwrite("small_positives/" + filename, img2)

def edgedetect (channel):
    sobelx = cv2.Sobel(channel, cv2.CV_16S, 1, 0, borderType=cv2.BORDER_REPLICATE, ksize=5)
    sobely = cv2.Sobel(channel, cv2.CV_16S, 0, 1, borderType=cv2.BORDER_REPLICATE, ksize=5)
    sobel = np.hypot(sobelx, sobely)
    return sobel

def findSignificantContours (img, sobel_8u):
    image, contours, heirarchy = cv2.findContours(sobel_8u, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find level 1 contours
    level1 = []
    for i, tupl in enumerate(heirarchy[0]):
        # Each array is in format (Next, Prev, First child, Parent)
        # Filter the ones without parent
        if tupl[3] == -1:
            tupl = np.insert(tupl, 0, [i])
            level1.append(tupl)

    # From among them, find the contours with large surface area.
    significant = []
    tooSmall = sobel_8u.size * 5 / 100 # If contour isn't covering 5% of total area of image then it probably is too small
    for tupl in level1:
        contour = contours[tupl[0]];
        area = cv2.contourArea(contour)
        if area > tooSmall:
            # cv2.drawContours(img, [contour], 0, (0,255,0),2, cv2.LINE_AA, maxLevel=1)
            significant.append([contour, area])

    significant.sort(key=lambda x: x[1])
    return [x[0] for x in significant]

def remove_background(filename):
    img = cv2.imread("raw_positives/"+ filename)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    edgeImg = np.max( np.array([ edgedetect(blurred[:,:, 0]), edgedetect(blurred[:,:, 1]), edgedetect(blurred[:,:, 2]) ]), axis=0 )
    mean = np.mean(edgeImg);
    # Zero any value that is less than mean. This reduces a lot of noise.
    edgeImg[edgeImg <= mean] = 0;
    edgeImg[edgeImg > 255] = 255

    edgeImg_8u = np.asarray(edgeImg, np.uint8)
    # Find contours
    significant = findSignificantContours(img, edgeImg_8u)

    # Mask
    mask = edgeImg.copy()
    mask[mask > 0] = 0
    cv2.fillPoly(mask, significant, 255)
    # Invert mask
    mask = np.logical_not(mask)

    #Finally remove the background
    img[mask] = 255;

    return img

def create_names(filename):

    line = filename +' 1 0 0 707 707\n'
    with open('positive-clean.txt','a') as f:
        f.write(line)

if __name__ == "__main__":
    # if not os.path.exists("positives"):
    #     os.makedirs("positives")
    #
    # for name in os.listdir('raw_positives'):
    #     process(name, remove_background(name))
    #     break


    for name in os.listdir('positives'):
        resize(name)
        create_names(name)
