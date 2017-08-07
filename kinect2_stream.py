# This does not use ROS

# coding: utf-8

import numpy as np
import cv2
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel

from pylibfreenect2 import CpuPacketPipeline
pipeline = CpuPacketPipeline()
print("Packet pipeline:", type(pipeline).__name__)

def sift(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray, None)
    return cv2.drawKeypoints(gray,kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def fast(img):
    # Initiate FAST object with default values
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fast = cv2.FastFeatureDetector_create()

    # find and draw the keypoints)
    kp = fast.detect(img,None)
    # get all the kp coordinates
    pts = [kp[idx].pt for idx in range(len(kp))]
    print pts
    sys.exit(1)
    return cv2.drawKeypoints(gray, kp, img, color=(255,0,0))


# Create and set logger
logger = createConsoleLogger(LoggerLevel.Debug)
setGlobalLogger(logger)
fn = Freenect2()
num_devices = fn.enumerateDevices()
if num_devices == 0:
    print("No device connected!")
    sys.exit(1)

serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial, pipeline=pipeline)

listener = SyncMultiFrameListener(
    FrameType.Color | FrameType.Ir | FrameType.Depth)

# Register listeners
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)

device.start()


# NOTE: must be called after device.start()
registration = Registration(device.getIrCameraParams(), device.getColorCameraParams())

undistorted = Frame(512, 424, 4)
registered = Frame(512, 424, 4)

while True:
    frames = listener.waitForNewFrame()

    color = frames["color"]
    depth = frames["depth"]

    registration.apply(color, depth, undistorted, registered)

    color = color.asarray()

    color_fast = fast(color)


    cv2.imshow("color", color_fast)


    listener.release(frames)

    key = cv2.waitKey(delay=1)
    if key == ord('q'):
        break

device.stop()
device.close()

sys.exit(0)
