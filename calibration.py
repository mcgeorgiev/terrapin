import cv2
import numpy as np
import sys,time
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class Camera:
    def __init__(self, sensor):
        self.image = None
        self.hsv = None
        self.bridge = CvBridge()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        rospy.init_node('calib', anonymous=True)

        if sensor == "kinect2":
            topic = "/kinect2/qhd/image_color_rect"
        elif sensor == "zed":
            topic = "/rgb/image_rect_color"
        rospy.Subscriber(topic, Image, self.callback)

    def callback(self, image_data):
        self.image = self.bridge.imgmsg_to_cv2(image_data, "bgr8")
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

    def record(self):
        average_color = None
        while not rospy.is_shutdown():
            frame = self.image
            cv2.rectangle(frame,((self.image.shape[1])/2-100,(self.image.shape[0])/2-100),((self.image.shape[1])/2+100,(self.image.shape[0])/2+100),(0,200,0),3)
            roi = self.hsv[(self.image.shape[0])/2-100:(self.image.shape[0])/2+100,(self.image.shape[1])/2-100:(self.image.shape[1])/2+100 ]

            average_color_per_row = np.average(roi, axis=0)
            average_color = np.average(average_color_per_row, axis=0)

            cv2.putText(frame, "Press 'q' to mask out the selected area based on hsv.",(50,50), self.font, 1,(255,0,0),2 ,cv2.LINE_AA)
            cv2.putText(frame, "HSV: " + str(average_color),((self.image.shape[1])/2-300,(self.image.shape[0])/2-150), self.font, 1,(200,0,200),2 ,cv2.LINE_AA)


            cv2.imshow("Calibration Tool", frame)
            key = cv2.waitKey(delay=1)

            if(cv2.waitKey(10)==ord('p')):
                print "Saving", average_color
                break
            if key == ord('q'):
                break

        with open("calib_hsv.txt", "a") as f:
            f.write(str(list(average_color))+"\n")




if __name__ == "__main__":
    if len(sys.argv) == 0:
        print "Please enter a camera as an argument"
        sys.exit()
    if sys.argv[1] not in ["zed", "kinect2"]:
        print "Please enter a valid camera type: zed, kinect2"
        sys.exit()
    r = Camera(sys.argv[1])
    time.sleep(5)
    r.record()
    cv2.destroyAllWindows()
