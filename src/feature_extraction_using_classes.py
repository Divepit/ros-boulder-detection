#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PointStamped
from grid_map_msgs.msg import GridMap
import numpy as np
from sklearn.cluster import DBSCAN

class Rock:
    def __init__(self, x_position, y_position, z_position):
        self.creation_time = rospy.Time.now()
        self.latest_update = self.creation_time
        self.x_position = x_position
        self.y_position = y_position
        self.z_position = z_position
        self.confirmed_rock = False
        self.time_until_rock_is_confirmed = rospy.get_param(rospy.get_name() + '/time_until_rock_is_confirmed')

    def get_position(self):
        return (self.x_position,self.y_position,self.z_position)
    
    def get_x(self):
        return self.x_position
    def get_y(self):
        return self.y_position
    def get_z(self):
        return self.z_position
    
    def get_confidence(self):
        return self.confidence
    
    def get_creation_time(self):
        return self.creation_time
    
    def get_age_sec(self):
        return (rospy.Time.now() - self.creation_time).to_sec()
    
    def get_time_since_last_updace_sec(self):
        return (rospy.Time.now() - self.latest_update).to_sec()
    
    def confirm_rock(self):
        self.confirmed_rock = True

    def is_confirmed(self):
        return self.confirmed_rock

    def update_position(self, new_rock):
        self.latest_update = rospy.Time.now()

        if self.is_confirmed():
            return

        self.x_position = (self.x_position + new_rock.get_x())/2
        self.y_position = (self.y_position + new_rock.get_y())/2
        self.z_position = (self.z_position + new_rock.get_z())/2

        if self.get_age_sec() > self.time_until_rock_is_confirmed:
            self.confirm_rock()

class RockMap:
    def __init__(self):
        self.all_markers = None
        self.active_rocks = []
        self.frame_id = ''
        self.resolution = 0.0
        self.length_x = 0.0
        self.length_y = 0.0
        self.position_x = 0
        self.position_y = 0
        self.elevation_data = None




    def run(self):
        rospy.init_node('rock_detector', anonymous=True)
        rospy.set_param('/use_sim_time', True)

        rospy.Subscriber("/elevation_mapping/elevation_map_raw", GridMap, self.callback)
        self.marker_pub = rospy.Publisher("/rock_markers", MarkerArray, queue_size=1)
        self.raw_marker_pub = rospy.Publisher("/raw_rock_points", MarkerArray, queue_size=1)

        rospy.spin()

    def callback(self, grid_map_msg):

        elevation_layer = grid_map_msg.layers.index('elevation')
        self.frame_id = grid_map_msg.info.header.frame_id
        self.resolution = float(grid_map_msg.info.resolution)
        self.length_x = float(grid_map_msg.info.length_x)
        self.length_y = float(grid_map_msg.info.length_y)
        self.position_x = float(grid_map_msg.info.pose.position.x)
        self.position_y = float(grid_map_msg.info.pose.position.y)
        data_dim_x = int(np.ceil(self.length_x / self.resolution))
        data_dim_y = int(np.ceil(self.length_y / self.resolution))
        self.elevation_data = np.array(grid_map_msg.data[elevation_layer].data, dtype=np.float32).reshape(data_dim_x, data_dim_y)
        

        # raw_points = self.extract_raw_points()
        # self.publish_raw_points(raw_points)

        self.sample_active_rocks()

        definitive_rocks = [rock for rock in self.active_rocks if rock.is_confirmed()]

        self.publish_rock_markers(definitive_rocks)
        self.clear_old_rocks()

    def clear_old_rocks(self):
        time_until_rock_is_removed = rospy.get_param(rospy.get_name() + '/time_until_rock_is_removed')
        for rock in self.active_rocks:
            if not rock.is_confirmed() and (rock.get_time_since_last_updace_sec() > time_until_rock_is_removed):
                self.active_rocks.remove(rock)

    
    def extract_raw_points(self):
        percentage_of_mean_elevation = rospy.get_param(rospy.get_name() + '/percentage_of_mean_elevation')
        mean_elevation = np.nanmean(self.elevation_data)
        rock_mask = (self.elevation_data > mean_elevation*percentage_of_mean_elevation)
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
        
        return np.array(points)

    def is_within_radius(self, new_rock, radius):
        if self.active_rocks is not None:
            for rock in self.active_rocks:
                if calculate_distance(new_rock.get_position(), rock.get_position()) < radius:
                    return rock
        return False

    def sample_active_rocks(self):
        dbscan_eps = rospy.get_param(rospy.get_name() + '/DBSCAN_eps')
        dbscan_min_samples = rospy.get_param(rospy.get_name() + '/DBSCAN_min_samples')
        same_rock_radius = rospy.get_param(rospy.get_name() + '/radius_to_appoint_to_same_rock')

        points = self.extract_raw_points()
        min_points = rospy.get_param(rospy.get_name() + '/minimum_points_to_qualify_as_rock')
        max_points = rospy.get_param(rospy.get_name() + '/maximum_points_to_qualify_as_rock')
        
        clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(points)
        labels = clustering.labels_
        
        unique_labels = set(labels)

        for label in unique_labels:
            if label != -1:
                class_member_mask = (labels == label)
                class_points = points[class_member_mask]
                if min_points <= len(class_points) <= max_points:
                    new_centroid = np.mean(class_points, axis=0)
                    new_rock = Rock(new_centroid[0],new_centroid[1],0)
                    rock_in_proximity = self.is_within_radius(new_rock, same_rock_radius)
                    if not rock_in_proximity:
                        self.active_rocks.append(new_rock)
                    else:
                        rock_in_proximity.update_position(new_rock)
        

    # def publish_raw_points(self, points):
    #     raw_marker_array = MarkerArray()
    #     current_time = rospy.Time.now()
        
    #     for i, point in enumerate(points):
    #         marker = Marker()
    #         marker.header.stamp = current_time
    #         marker.type = Marker.SPHERE
    #         marker.header.frame_id = self.frame_id
    #         marker.scale.x = 0.1
    #         marker.scale.y = 0.1
    #         marker.scale.z = 0.1
    #         marker.color.r = 1.0
    #         marker.color.g = 0.0
    #         marker.color.b = 0.0
    #         marker.color.a = 1.0
    #         marker.pose.orientation.w = 1.0
    #         marker.lifetime = rospy.Duration()
            
    #         marker.pose.position.x = point[0]
    #         marker.pose.position.y = point[1]
    #         marker.pose.position.z = point[2]
    #         marker.id = i
            
    #         raw_marker_array.markers.append(marker)
        
    #     self.raw_marker_pub.publish(raw_marker_array)

    def publish_rock_markers(self, rocks):
        if self.all_markers is None:
            self.all_markers = MarkerArray()
        current_time = rospy.Time.now()
        current_marker_count = len(self.all_markers.markers)
        centroid_count = len(rocks)
        if centroid_count > current_marker_count:
            for i in range(current_marker_count, centroid_count):
                marker = Marker()
                marker.header.frame_id = self.frame_id
                marker.type = Marker.CUBE
                marker.scale.x = 0.6
                marker.scale.y = 0.6
                marker.scale.z = 0.6
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0
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
            self.all_markers.markers[i].id = i

        self.marker_pub.publish(self.all_markers)

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


if __name__ == '__main__':
    node = RockMap()
    node.run()