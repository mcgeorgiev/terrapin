# coding: utf-8

import numpy as np
import cv2
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel

from pylibfreenect2 import CpuPacketPipeline

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def projection():

    pipeline = CpuPacketPipeline()
    print("Packet pipeline:", type(pipeline).__name__)

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
    registration = Registration(device.getIrCameraParams(),
                                device.getColorCameraParams())

    undistorted = Frame(512, 424, 4)
    registered = Frame(512, 424, 4)

    # Optinal parameters for registration
    # set True if you need
    need_bigdepth = True
    need_color_depth_map = True

    bigdepth = Frame(1920, 1082, 4) if need_bigdepth else None
    color_depth_map = np.zeros((424, 512),  np.int32).ravel() \
        if need_color_depth_map else None


    color_pub = rospy.Publisher('color', Image, queue_size=10)
    depth_pub = rospy.Publisher('depth', Image, queue_size=10)

    rospy.init_node('projection', anonymous=True)
    rate = rospy.Rate(10) # 10hz

    while not rospy.is_shutdown():
        frames = listener.waitForNewFrame()

        color = frames["color"]
        ir = frames["ir"]
        depth = frames["depth"]

        registration.apply(color, depth, undistorted, registered,
                           bigdepth=bigdepth,
                           color_depth_map=color_depth_map)

        color_image = CvBridge().cv2_to_imgmsg(color.asarray())
        color_pub.publish(color_image)


        depth_image = CvBridge().cv2_to_imgmsg(bigdepth.asarray(np.float32))
        depth_pub.publish(depth_image)

        rate.sleep()

        listener.release(frames)

        key = cv2.waitKey(delay=1)
        if key == ord('q'):
            break

    device.stop()
    device.close()
    sys.exit(0)

try:
    projection()
except rospy.ROSInterruptException:
    pass
