#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Int32MultiArray
from geometry_msgs.msg import PointStamped
from grid_map_msgs.msg import GridMap
import numpy as np
from sklearn.cluster import DBSCAN
from rock_class import Rock


class RockMap:
    def __init__(self):

        rospy.init_node('rock_detector', anonymous=True)
        # rospy.set_param('/use_sim_time', True)

        rospy.Subscriber("/elevation_mapping/elevation_map_raw", GridMap, self.callback)
        rospy.Subscriber("/confirmed_rock_ids", Int32MultiArray, self.confirmed_rocks_callback)
        self.marker_pub = rospy.Publisher("/rock_markers", MarkerArray, queue_size=1)
        self.raw_marker_pub = rospy.Publisher("/raw_rock_points", MarkerArray, queue_size=1)

        self.all_markers = None
        self.active_rocks = []
        self.raw_points = []
        self.last_raw_points_sample_time = rospy.Time.now()
        self.frame_id = None
        self.elevation_layer = None
        self.resolution = 0.0
        self.length_x = 0.0
        self.length_y = 0.0
        self.position_x = 0
        self.position_y = 0
        self.elevation_data = None
        self.next_id = 0

        self.dbscan_eps = rospy.get_param(rospy.get_name() + '/DBSCAN_eps')
        self.dbscan_min_samples = rospy.get_param(rospy.get_name() + '/DBSCAN_min_samples')
        self.same_rock_radius = rospy.get_param(rospy.get_name() + '/radius_to_appoint_to_same_rock')
        self.min_points = rospy.get_param(rospy.get_name() + '/minimum_points_to_qualify_as_rock')
        self.max_points = rospy.get_param(rospy.get_name() + '/maximum_points_to_qualify_as_rock')
        self.percentage_of_mean_elevation = rospy.get_param(rospy.get_name() + '/percentage_of_mean_elevation')
        self.time_until_rock_is_removed = rospy.get_param(rospy.get_name() + '/time_until_rock_is_removed')
        self.do_publish_raw_points = rospy.get_param(rospy.get_name() + '/do_publish_raw_points')
        self.raw_points_sample_rate = rospy.get_param(rospy.get_name() + '/raw_points_sample_rate')

        rospy.spin()


    def callback(self, grid_map_msg):
        
        if not self.elevation_layer:
            self.elevation_layer = grid_map_msg.layers.index('elevation')
        if not self.frame_id:
            self.frame_id = grid_map_msg.info.header.frame_id
        
        self.resolution = float(grid_map_msg.info.resolution)
        self.length_x = float(grid_map_msg.info.length_x)
        self.length_y = float(grid_map_msg.info.length_y)
        self.position_x = float(grid_map_msg.info.pose.position.x)
        self.position_y = float(grid_map_msg.info.pose.position.y)

        data_dim_x = int(np.ceil(self.length_x / self.resolution))
        data_dim_y = int(np.ceil(self.length_y / self.resolution))

        self.elevation_data = np.array(grid_map_msg.data[self.elevation_layer].data, dtype=np.float32).reshape(data_dim_x, data_dim_y)
        
        if self.do_publish_raw_points:
            if len(self.raw_points) > 0:
                self.publish_raw_points(self.raw_points)

        self.sample_active_rocks()

        definitive_rocks = [rock for rock in self.active_rocks if rock.is_confirmed()]

        self.publish_rock_markers(definitive_rocks)
        self.clear_old_rocks()

    def confirmed_rocks_callback(self, msg):
        # compare ids of active rocks to ids in msg
        if len(msg.data) > 0:
            for rock in self.active_rocks:
                if rock.id in msg.data:
                    rock.ml_confirm_rock()
            

    def clear_old_rocks(self):
        for rock in self.active_rocks:
            if not rock.is_confirmed() and (rock.get_time_since_last_updace_sec() > self.time_until_rock_is_removed):
                self.active_rocks.remove(rock)

    
    def extract_raw_points(self):
        current_time = rospy.Time.now()

        if current_time < self.last_raw_points_sample_time:
            rospy.logwarn('Time reset detected. Resetting last_raw_points_sample_time .')
            self.last_raw_points_sample_time = current_time
            self.raw_points = []

        # rospy.loginfo(current_time)
        # rospy.loginfo((self.last_raw_points_sample_time).to_sec())
        # rospy.loginfo(1/self.raw_points_sample_rate)
        # rospy.loginfo((current_time - self.last_raw_points_sample_time).to_sec())
        if (current_time - self.last_raw_points_sample_time).to_sec() > 1/self.raw_points_sample_rate:
            min_value = np.nanmin(self.elevation_data)
            elevation_data = self.elevation_data

            if min_value < 0:
                elevation_data = self.elevation_data + np.abs(min_value)
            max_value = np.nanmax(elevation_data)
            mean_elevation = np.nanmean(elevation_data)
    

            # rospy.loginfo("Mean elevation: %f", mean_elevation)
            # rospy.loginfo("Max elevation: %f", max_value)
            # rospy.loginfo("Min elevation: %f", min_value)


            sanitized_data = np.nan_to_num(elevation_data, nan=0, posinf=max_value, neginf=0)
            rock_mask = (np.abs(sanitized_data) > np.abs(mean_elevation)*self.percentage_of_mean_elevation)

            y_indices, x_indices = np.where(rock_mask)
            points = []

            for x_index, y_index in zip(x_indices, y_indices):
                x = -x_index*self.resolution+self.length_x/2+self.position_x
                y = -y_index*self.resolution+self.length_y/2+self.position_y
                
                # Convert from grid coordinates to world coordinates using the pose
                point = PointStamped()
                point.header.frame_id = self.frame_id
                point.header.stamp = rospy.Time.now()
                point.point.x = x
                point.point.y = y
                point.point.z = 0  # Adjust the z-coordinate as necessary
                points.append([point.point.x, point.point.y, point.point.z])
                self.raw_points = points
                self.last_raw_points_sample_time = current_time
            
            return np.array(points)
        return np.array(self.raw_points)

    def is_within_radius(self, new_rock, radius):
        if self.active_rocks is not None:
            for rock in self.active_rocks:
                if calculate_distance(new_rock.get_position(), rock.get_position()) < radius:
                    return rock
        return False

    def sample_active_rocks(self):
        points = self.extract_raw_points()
        if len(points) == 0:
            return
        clustering = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit(points)
        labels = clustering.labels_
        
        unique_labels = set(labels)

        for label in unique_labels:
            if label != -1:
                class_member_mask = (labels == label)
                class_points = points[class_member_mask]
                if self.min_points <= len(class_points) <= self.max_points:
                    new_centroid = np.mean(class_points, axis=0)
                    new_rock = Rock(new_centroid[0],new_centroid[1],0,self.next_id)
                    self.next_id += 1
                    rock_in_proximity = self.is_within_radius(new_rock, self.same_rock_radius)
                    # rospy.loginfo("Min distance between points: %f", calculate_min_distance(class_points))
                    # rospy.loginfo("Number of points in rock: %d", len(class_points))
                    if not rock_in_proximity:
                        self.active_rocks.append(new_rock)
                    else:
                        rock_in_proximity.update_position(new_rock)
        

    def publish_raw_points(self, points):
        raw_marker_array = MarkerArray()
        current_time = rospy.Time.now()
        
        for i, point in enumerate(points):
            marker = Marker()
            marker.header.stamp = current_time
            marker.type = Marker.SPHERE
            marker.header.frame_id = self.frame_id
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.pose.orientation.w = 1.0
            marker.lifetime = rospy.Duration()
            
            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker.pose.position.z = point[2]
            marker.id = i
            
            raw_marker_array.markers.append(marker)
                
        self.raw_marker_pub.publish(raw_marker_array)

    def publish_rock_markers(self, rocks):
        if self.all_markers is None:
            self.all_markers = MarkerArray()
        current_time = rospy.Time.now()
        current_marker_count = len(self.all_markers.markers)
        centroid_count = len(rocks)
        if centroid_count > current_marker_count:
            for i in range(current_marker_count, centroid_count):
                marker = Marker()
                marker.id = rocks[i].get_id()
                marker.header.frame_id = self.frame_id
                marker.type = Marker.CUBE
                marker.scale.x = 0.6
                marker.scale.y = 0.6
                marker.scale.z = 0.6
                marker.color.a = 1.0
                marker.color.r = 0.0
                marker.color.b = 0.0
                marker.color.g = 1.0
                marker.pose.orientation.w = 1.0
                marker.lifetime = rospy.Duration()
                self.all_markers.markers.append(marker)
        elif centroid_count < current_marker_count:
            del self.all_markers.markers[centroid_count:]
        for i, rock in enumerate(rocks):
            self.all_markers.markers[i].header.stamp = current_time
            self.all_markers.markers[i].pose.position.x = rock.get_x()
            self.all_markers.markers[i].pose.position.y = rock.get_y()
            self.all_markers.markers[i].pose.position.z = rock.get_z()
            self.all_markers.markers[i].id = rock.get_id()
            if rock.is_ml_confirmed():
                self.all_markers.markers[i].color.r = 0.5
                self.all_markers.markers[i].color.b = 0.5
                self.all_markers.markers[i].color.g = 0.0

        self.marker_pub.publish(self.all_markers)

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def calculate_min_distance(points):
    min_distance = float('inf')
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = calculate_distance(points[i], points[j])
            if dist < min_distance:
                min_distance = dist
    return min_distance

if __name__ == '__main__':
    node = RockMap()
