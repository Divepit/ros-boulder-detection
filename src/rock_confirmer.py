#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import tf2_ros
import tf2_geometry_msgs
import math
from std_msgs.msg import Bool
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PoseStamped


class RockMap:
    def __init__(self):
        

        self.robot_pose = None
        self.looking_at_rock = False
        self.max_distance = 5
        self.angle_range = math.pi
        self.base_frame = "base"
        self.confirmed_rocks = []




    def run(self):
        rospy.init_node('rock_confirmer', anonymous=True)
        rospy.set_param('/use_sim_time', True)

        rospy.Subscriber("/rock_markers", MarkerArray, self.rock_marker_callback)
        rospy.Subscriber("/rock_in_sight", Bool, self.rock_in_sight_callback)
        self.confirmed_rocks_pub = rospy.Publisher("/confirmed_rocks", MarkerArray, queue_size=1)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.spin()

    def transform_to_base_frame(self, rock_marker):
        try:
            marker_pose = PoseStamped()
            marker_pose.header = rock_marker.header
            marker_pose.pose = rock_marker.pose

            transformed_pose = self.tf_buffer.transform(marker_pose, 'base', rospy.Duration(1.0))
            return transformed_pose

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Failed to transform rock marker to base frame")
            return None

    def rock_in_sight_callback(self, msg):
        self.looking_at_rock = msg.data
        
    def rock_marker_callback(self, msg):

        try:
            marker_in_base_frame = self.transform_to_base_frame(msg.markers[0])
        except IndexError:
            return

        if marker_in_base_frame is None:
            return

        distance = math.sqrt(marker_in_base_frame.pose.position.x**2 + marker_in_base_frame.pose.position.y**2)
        angle = math.atan2(marker_in_base_frame.pose.position.y, marker_in_base_frame.pose.position.x)

        if distance < self.max_distance and abs(angle) < self.angle_range/2 and self.looking_at_rock:
            self.confirmed_rocks.append(marker_in_base_frame)
        
        self.publish_confirmed_rocks()


    def publish_confirmed_rocks(self):
        marker_array = MarkerArray()
        current_time = rospy.Time.now()

        for i, rock in enumerate(self.confirmed_rocks):
            marker = Marker()
            marker.header.stamp = current_time
            marker.header.frame_id = self.base_frame
            marker.type = Marker.CUBE
            marker.scale.x = 0.6
            marker.scale.y = 0.6
            marker.scale.z = 0.6
            marker.color.r = 0.5
            marker.color.g = 0.0
            marker.color.b = 0.5
            marker.color.a = 1.0
            marker.pose.orientation.w = 1.0
            marker.lifetime = rospy.Duration()
            marker.pose = rock.pose
            marker.id = i
            marker_array.markers.append(marker)

        self.confirmed_rocks_pub.publish(marker_array)


if __name__ == '__main__':
    node = RockMap()
    node.run()