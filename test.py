# coding: utf-8

# An example using startStreams

import numpy as np
import cv2
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame


from pylibfreenect2 import CpuPacketPipeline
pipeline = CpuPacketPipeline()
print("Packet pipeline:", type(pipeline).__name__)

enable_rgb = True
enable_depth = True

fn = Freenect2()
num_devices = fn.enumerateDevices()
if num_devices == 0:
    print("No device connected!")
    sys.exit(1)

serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial, pipeline=pipeline)

types = 0
if enable_rgb:
    types |= FrameType.Color
if enable_depth:
    types |= (FrameType.Ir | FrameType.Depth)
listener = SyncMultiFrameListener(types)

# Register listeners
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)

if enable_rgb and enable_depth:
    device.start()
else:
    device.startStreams(rgb=enable_rgb, depth=enable_depth)

# NOTE: must be called after device.start()
if enable_depth:
    registration = Registration(device.getIrCameraParams(),
                                device.getColorCameraParams())

undistorted = Frame(512, 424, 4)
registered = Frame(512, 424, 4)
color = Frame(512, 424, 4)

#capture
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# depth_out = cv2.VideoWriter('depth.avi', fourcc,10.0, (512,424))
color_out = cv2.VideoWriter('color_20.avi', fourcc ,2.0, (1920,1080))



while True:
    frames = listener.waitForNewFrame()

    if enable_rgb:
        color = frames["color"]
    if enable_depth:
        ir = frames["ir"]
        depth = frames["depth"]

    if enable_rgb and enable_depth:
        registration.apply(color, depth, undistorted, registered)
    elif enable_depth:
        registration.undistortDepth(depth, undistorted)

    # if enable_depth:
        # cv2.imshow("ir", ir.asarray() / 65535.)
        # d_array = (depth.asarray() / 4500.)
        # print d_array.shape
        # depth_out.write(d_array)
        # cv2.imshow("depth", depth.asarray() / 4500.)
        # cv2.imshow("undistorted", undistorted.asarray(np.float32) / 4500.)
    if enable_rgb:
        color_array = color.asarray()
        color_out.write(color_array)
        cv2.imshow("color", color_array)
    # if enable_rgb and enable_depth:
    #     cv2.imshow("registered", registered.asarray(np.uint8))

    listener.release(frames)

    key = cv2.waitKey(delay=1)
    if key == ord('q'):
        break

device.stop()
device.close()
# depth_out.release()
color_out.release()
sys.exit(0)
