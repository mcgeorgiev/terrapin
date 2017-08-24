#!/usr/bin/env python
import rospy
import tf2_ros
from sensor_msgs.msg import PointCloud2
import PyKDL
from PyKDL import Vector
import PyKDL
import sensor_msgs.point_cloud2  as pc2
from geometry_msgs.msg import PointStamped, Vector3Stamped
import numpy as np
import tf2_geometry_msgs
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
class Marker:
    def __init__(self):
        rospy.Subscriber('/camera/depth/points', PointCloud2, self.point_cloud_callback)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.translation = None
        self.rotation = None
        self.transform = None
        self.markerArray = MarkerArray()
        self.count = 0
        self.MARKERS_MAX = 1
        self.point_3d_array = None
        self.publisher = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)

    def get_xyz(self, x, y):
        # print self.point_3d_array
        return self.point_3d_array[y][x]

    def to_msg_vector(self, vector):
        msg = PointStamped()
        msg.header.frame_id = "camera_depth_optical_frame"
        msg.header.stamp = rospy.Time(0)
        msg.point.x = vector[0]
        msg.point.y = vector[1]
        msg.point.z = vector[2]
        return msg

    def point_cloud_callback(self, msg):
        point_cloud = msg

        # print "getting point data"
        point_list = []
        for p in pc2.read_points(point_cloud, field_names = ("x", "y", "z")):
            point_list.append((p[0],p[1],p[2]))
        point_array = np.array(point_list)
        self.point_3d_array = np.reshape(point_array, (480,640,3))

        self.transform = self.tf_buffer.lookup_transform("map",
                                                        msg.header.frame_id,
                                                        rospy.Time(0),
                                                        rospy.Duration(10))
        self.translation = self.transform.transform.translation
        self.rotation = self.transform.transform.rotation


    def mark(self, x, y, z):
        marker = Marker()
        marker.header.frame_id = "/map"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z

        # We add the new marker to the MarkerArray, removing the oldest
        # marker from it when necessary
        # if(self.count > self.MARKERS_MAX):
        #     self.markerArray.markers.pop(0)

        self.markerArray.markers.append(marker)

        # Renumber the marker IDs
        id = 0
        for m in self.markerArray.markers:
            m.id = id
            id += 1

        # Publish the MarkerArray
        self.publisher.publish(self.markerArray)

        self.count += 1

    def transform_to_kdl(self, t):
        return PyKDL.Frame(PyKDL.Rotation.Quaternion(t.transform.rotation.x, t.transform.rotation.y,
                                                  t.transform.rotation.z, t.transform.rotation.w),
                        PyKDL.Vector(t.transform.translation.x,
                                     t.transform.translation.y,
                                     t.transform.translation.z))

    def do_transform_vector3(self,vector3, transform):
        p = self.transform_to_kdl(transform) * PyKDL.Vector(vector3.point.x, vector3.point.y, vector3.point.z)
        res = Vector3Stamped()
        res.vector.x = p[0]
        res.vector.y = p[1]
        res.vector.z = p[2]
        res.header = transform.header
        return res

    def listen(self):
        while  not rospy.is_shutdown():
            try:
                x,y,z = self.get_xyz(320,240)
                vec = self.to_msg_vector(Vector(x,y,z))
                if self.transform:
                    transformed_vec = self.do_transform_vector3(vec, self.transform)
                    x,y,z = transformed_vec.vector.x, transformed_vec.vector.y, transformed_vec.vector.z
                    self.mark(x,y,z)
                    print "published marker"
            except Exception as e:
                pass

m = Marker()
m.listen()
