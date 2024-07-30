#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import tf2_ros
import tf2_geometry_msgs
import math
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Int32MultiArray
from rock_segmentation.msg import RockVectors
from geometry_msgs.msg import Vector3Stamped, PointStamped, Vector3

class RockConfirmer:
    def __init__(self):
        rospy.init_node('rock_confirmer', anonymous=True)

        self.camera_frame = None
        self.confirmed_rock_ids = []
        self.tolerance_sphere_radius = 0.8
        self.rock_markers = []
        # self.current_rays = MarkerArray()
        
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.trans = None
        
        rospy.Subscriber("/rock_markers", MarkerArray, self.rock_marker_callback)
        rospy.Subscriber("/rock_rays", RockVectors, self.ray_callback)
        self.confirmed_rocks_pub = rospy.Publisher("/confirmed_rock_ids", Int32MultiArray, queue_size=1)
        self.ray_markers_pub = rospy.Publisher("/ray_markers", MarkerArray, queue_size=1)
        
        rospy.spin()

    def get_transforms(self):
        if self.camera_frame is None:
            return
        else:
            try:
                self.trans = self.tfBuffer.lookup_transform("map_o3d_localization_manager", self.camera_frame, rospy.Time(0))
            except:
                rospy.logwarn("Could not get transform from base to map_o3d_localization_manager")

    def ray_callback(self, msg):
        if self.camera_frame is None:
            self.camera_frame = msg.header.frame_id
        if self.trans is None:
            return
        else: 
            for ray in msg.vectors:
                if ray is not None:
                    self.check_ray_intersection(ray)
            self.publish_confirmed_rock_ids()

    def check_ray_intersection(self,ray):
        for marker in self.rock_markers:
            if marker.id not in self.confirmed_rock_ids and self.ray_intersects_sphere(ray, marker.pose.position, self.tolerance_sphere_radius):
                self.confirmed_rock_ids.append(marker.id)
        

    def rock_marker_callback(self, msg):
        self.get_transforms()
        self.rock_markers = msg.markers

    def ray_intersects_sphere(self, ray, sphere_center, sphere_radius):
        # ray: Vector3
        # sphere_center: Point
        # sphere_radius: float
        # Returns: bool
        
        ray_origin = PointStamped()
        ray_origin.header.frame_id = self.camera_frame
        ray_origin.point.x = 0
        ray_origin.point.y = 0
        ray_origin.point.z = 0.1
        
        ray_direction = Vector3Stamped()
        ray_direction.header.frame_id = self.camera_frame
        ray_direction.vector = ray
        
        sphere_center_stamped = PointStamped()
        sphere_center_stamped.header.frame_id = "map_o3d_localization_manager"  # Assuming the sphere_center is already in the map_o3d_localization_manager frame
        sphere_center_stamped.point = sphere_center
        
        ray_origin_map_o3d_localization_manager = self.tfBuffer.transform(ray_origin, "map_o3d_localization_manager")
        ray_direction_map_o3d_localization_manager = self.tfBuffer.transform(ray_direction, "map_o3d_localization_manager")
        
        # Calculate vector from ray origin to sphere center
        oc = Vector3()
        oc.x = sphere_center.x - ray_origin_map_o3d_localization_manager.point.x
        oc.y = sphere_center.y - ray_origin_map_o3d_localization_manager.point.y
        oc.z = sphere_center.z - ray_origin_map_o3d_localization_manager.point.z
        
        # Calculate quadratic equation coefficients
        a = ray_direction_map_o3d_localization_manager.vector.x**2 + ray_direction_map_o3d_localization_manager.vector.y**2 + ray_direction_map_o3d_localization_manager.vector.z**2
        b = 2 * (oc.x * ray_direction_map_o3d_localization_manager.vector.x + oc.y * ray_direction_map_o3d_localization_manager.vector.y + oc.z * ray_direction_map_o3d_localization_manager.vector.z)
        c = oc.x**2 + oc.y**2 + oc.z**2 - sphere_radius**2
        
        # Calculate discriminant
        discriminant = b**2 - 4*a*c
        
        # Check if the ray intersects the sphere
            # Check if the ray intersects the sphere
        if discriminant < 0:
            return False

        # Calculate the intersection points
        t1 = (-b - math.sqrt(discriminant)) / (2 * a)
        t2 = (-b + math.sqrt(discriminant)) / (2 * a)

        # Check if at least one intersection point is in front of the camera (t > 0)
        if t1 < 0 or t2 < 0:
            return True
        else:
            return False


    def publish_confirmed_rock_ids(self):
        msg = Int32MultiArray()
        msg.data = self.confirmed_rock_ids
        self.confirmed_rocks_pub.publish(msg)

if __name__ == '__main__':
    node = RockConfirmer()