import cv2

def process(filename):
    img = cv2.imread(filename,1)
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img_filt = cv2.medianBlur(img_gray, 5)
    threshold = cv2.adaptiveThreshold(img_filt,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)
    img2, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    sortedContours = sorted(contours, key =cv2.contourArea, reverse = True)
    largest_contours = sortedContours[0:6]
    # cv2.drawContours(img, largest_contours, -1, (0,255,0), 1)

    contourCentres = {}
    contourNumber = 1
    for contour in largest_contours:
        x,y,w,h = cv2.boundingRect(contour)
        contourCentres[contourNumber]=[(x+w)/2,(y+h)/2, w,h]
        contourNumber+=1
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imwrite('test.png',img)



if __name__ == "__main__":
    process("chair.jpg")
