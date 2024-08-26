#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy

import numpy as np

from sklearn.cluster import DBSCAN


from std_msgs.msg import Int32MultiArray
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from grid_map_msgs.msg import GridMap
from rock_detection_msgs.msg import ObstacleList, Obstacle
from rock_detection_msgs.srv import SetObstacleType, SetObstacleLatLon

class ElevationObstacle:
    def __init__(self, x_position, y_position, z_position, id):
        self.obstacle_id = id
        self.creation_time = rospy.Time.now()
        
        self.time_until_obstacle_is_confirmed = rospy.get_param(rospy.get_name() + '/time_until_obstacle_is_confirmed')
        

        self.latest_update = self.creation_time
        self.x_position = x_position
        self.y_position = y_position
        self.z_position = z_position

        self.lat = 0
        self.lon = 0
        
        self.confirmed_obstacle = False
        
        self.obstacle_type = "obstacle"
        

    def get_position(self):
        return (self.x_position,self.y_position,self.z_position)
    
    def get_x(self):
        return self.x_position
    def get_y(self):
        return self.y_position
    def get_z(self):
        return self.z_position
    
    def get_id(self):
        return self.obstacle_id

    def get_confidence(self):
        return self.confidence
    
    def get_creation_time(self):
        return self.creation_time
    
    def get_age_sec(self):
        return (rospy.Time.now() - self.creation_time).to_sec()
    
    def get_time_since_last_updace_sec(self):
        return (rospy.Time.now() - self.latest_update).to_sec()
    
    def confirm_obstacle(self):
        self.confirmed_obstacle = True

    def is_confirmed(self):
        return self.confirmed_obstacle
    
    def set_obstacle_type(self, obstacle_type):
        self.obstacle_type = obstacle_type


    def update_position(self, new_obstacle):
        self.latest_update = rospy.Time.now()

        if self.is_confirmed():
            return

        self.x_position = (self.x_position + new_obstacle.get_x())/2
        self.y_position = (self.y_position + new_obstacle.get_y())/2
        self.z_position = (self.z_position + new_obstacle.get_z())/2

        if self.get_age_sec() > self.time_until_obstacle_is_confirmed:
            self.confirm_obstacle()


class ObstacleDetector:
    def __init__(self):
        rospy.init_node('obstacle_detector', anonymous=True)

        # Subscribers
        rospy.Subscriber("/elevation_mapping/elevation_map_raw", GridMap, self.elevation_map_callback)
        rospy.Service("/rock_detection/set_obstacle_type", SetObstacleType, self.set_obstacle_type)
        rospy.Service("/rock_detection/set_obstacle_lat_lon", SetObstacleLatLon, self.set_obstacle_lat_lon)

        # Publishers
        self.obstacle_list_pub = rospy.Publisher("/rock_detection/obstacle_list", ObstacleList, queue_size=1)
        self.obstacle_marker_pub = rospy.Publisher("/rock_detection/obstacle_markers", MarkerArray, queue_size=1)
        self.raw_marker_pub = rospy.Publisher("/rock_detection/raw_elevation_points", MarkerArray, queue_size=1)

        # Parameters

        self.dbscan_eps = rospy.get_param(rospy.get_name() + '/DBSCAN_eps')
        self.dbscan_min_samples = rospy.get_param(rospy.get_name() + '/DBSCAN_min_samples')
        self.same_obstacle_radius = rospy.get_param(rospy.get_name() + '/radius_to_appoint_to_same_obstacle')
        self.min_points = rospy.get_param(rospy.get_name() + '/minimum_points_to_qualify_as_obstacle')
        self.max_points = rospy.get_param(rospy.get_name() + '/maximum_points_to_qualify_as_obstacle')
        self.percentage_of_mean_elevation = rospy.get_param(rospy.get_name() + '/percentage_of_mean_elevation')

        self.do_publish_raw_points = rospy.get_param(rospy.get_name() + '/do_publish_raw_points')
        self.raw_points_sample_rate = rospy.get_param(rospy.get_name() + '/raw_points_sample_rate')

        self.frame_id = None
        
        self.elevation_data = None
        self.resolution = 0.0
        self.length_x = 0.0
        self.length_y = 0.0
        self.elevation_layer = None
        
        self.position_x = 0 #TODO: rename
        self.position_y = 0 #TODO: rename
        
        self.detected_obstacles = []
        self.raw_points = []
        self.obstacle_markers = None
        
        self.next_id = 0

        self.last_raw_points_sample_time = rospy.Time.now()

        rospy.spin()

    def set_obstacle_type(self, req):
        for obstacle in self.detected_obstacles:
            if obstacle.obstacle_id == req.obstacle_id:
                obstacle.set_obstacle_type(req.obstacle_type)
                rospy.loginfo('Obstacle with id {} set to type {}'.format(req.obstacle_id, req.obstacle_type))
                return True
            else:
                rospy.logwarn('Type setting: Obstacle with id {} not found'.format(req.obstacle_id))
        return False
    
    def set_obstacle_lat_lon(self, req):
        for obstacle in self.detected_obstacles:
            if obstacle.obstacle_id == req.obstacle_id:
                obstacle.lat = req.lat
                obstacle.lon = req.lon
                rospy.loginfo('Obstacle with id {} set to lat: {}, lon: {}'.format(req.obstacle_id, req.lat, req.lon))
                return True
            else:
                rospy.logwarn('LatLon setting: Obstacle with id {} not found'.format(req.obstacle_id))
        return False

    def elevation_map_callback(self, msg):
        if not self.elevation_layer:
            self.elevation_layer = msg.layers.index('elevation')
        if not self.frame_id:
            self.frame_id = msg.info.header.frame_id
        
        self.resolution = float(msg.info.resolution)
        self.length_x = float(msg.info.length_x)
        self.length_y = float(msg.info.length_y)
        self.position_x = float(msg.info.pose.position.x)
        self.position_y = float(msg.info.pose.position.y)

        data_dim_x = int(np.ceil(self.length_x / self.resolution))
        data_dim_y = int(np.ceil(self.length_y / self.resolution))

        self.elevation_data = np.array(msg.data[self.elevation_layer].data, dtype=np.float32).reshape(data_dim_x, data_dim_y)
        
        if self.do_publish_raw_points:
            if len(self.raw_points) > 0:
                self.publish_raw_points(self.raw_points)

        self.detect_obstacles()

        obstacles = [obstacle for obstacle in self.detected_obstacles if obstacle.is_confirmed()]

        self.publish_obstacles(obstacles)

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

    def detect_obstacles(self):
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
                    new_obstacle = ElevationObstacle(new_centroid[0],new_centroid[1],0, self.next_id)
                    self.next_id += 1
                    obstacle_in_proximity = self.is_within_radius(new_obstacle, self.same_obstacle_radius)
                    if not obstacle_in_proximity:
                        self.detected_obstacles.append(new_obstacle)
                    else:
                        obstacle_in_proximity.update_position(new_obstacle)

    def extract_raw_points(self):
        current_time = rospy.Time.now()

        if current_time < self.last_raw_points_sample_time:
            rospy.logwarn('Time reset detected. Resetting last_raw_points_sample_time .')
            self.last_raw_points_sample_time = current_time
            self.detected_obstacles = []
            self.obstacle_markers = None

        if (current_time - self.last_raw_points_sample_time).to_sec() > 1/self.raw_points_sample_rate:
            min_value = np.nanmin(self.elevation_data)
            elevation_data = self.elevation_data

            if min_value < 0:
                elevation_data = self.elevation_data + np.abs(min_value)
            max_value = np.nanmax(elevation_data)
            mean_elevation = np.nanmean(elevation_data)
            sanitized_data = np.nan_to_num(elevation_data, nan=0, posinf=max_value, neginf=0)
            obstacle_mask = (np.abs(sanitized_data) > np.abs(mean_elevation)*self.percentage_of_mean_elevation)

            y_indices, x_indices = np.where(obstacle_mask)
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
    
    def is_within_radius(self, new_obstacle, radius):
        if self.detected_obstacles is not None:
            for obstacle in self.detected_obstacles:
                if self.calculate_distance(new_obstacle.get_position(), obstacle.get_position()) < radius:
                    return obstacle
        return False
    
    def calculate_distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def publish_obstacles(self, obstacles):
        obstacle_list_message = ObstacleList()
        obstacle_list_message.header.stamp = rospy.Time.now()
        obstacle_list_message.header.frame_id = self.frame_id
        # if self.obstacle_markers is None:
        self.obstacle_markers = MarkerArray()
        current_time = rospy.Time.now()
        # current_marker_count = len(self.obstacle_markers.markers)
        # centroid_count = len(obstacles)
        # if centroid_count > current_marker_count:
            # for i in range(current_marker_count, centroid_count):
        # elif centroid_count < current_marker_count:
        #     del self.obstacle_markers.markers[centroid_count:]
        for i, obstacle in enumerate(obstacles):
            marker = Marker()
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
            marker.header.stamp = current_time
            marker.pose.position.x = obstacle.get_x()
            marker.pose.position.y = obstacle.get_y()
            marker.pose.position.z = obstacle.get_z()
            marker.id = obstacle.get_id()

            if obstacle.obstacle_type == 'rock':
                marker.color.r = 0.5
                marker.color.b = 0.5
                marker.color.g = 0.0
            
            self.obstacle_markers.markers.append(marker)

            new_obstacle = Obstacle()
            new_obstacle.header.stamp = current_time
            new_obstacle.header.frame_id = self.frame_id
            new_obstacle.type = obstacle.obstacle_type
            new_obstacle.obstacle_id = obstacle.obstacle_id
            new_obstacle.lat = obstacle.lat
            new_obstacle.lon = obstacle.lon
            new_obstacle.x = obstacle.get_x()
            new_obstacle.y = obstacle.get_y()
            new_obstacle.z = obstacle.get_z()

            obstacle_list_message.obstacles.append(new_obstacle)
            

        self.obstacle_list_pub.publish(obstacle_list_message)

        self.obstacle_marker_pub.publish(self.obstacle_markers)
    
if __name__ == '__main__':
    node = ObstacleDetector()