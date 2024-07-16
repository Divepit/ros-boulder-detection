#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from visualization_msgs.msg import MarkerArray, Marker
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray
import numpy as np
from sklearn.cluster import DBSCAN

all_markers = None
latest_centroids = []
frame_id_global = ''
resolution_global = 0.0
length_x_global = 0.0
length_y_global = 0.0
position_x_global = 0
position_y_global = 0

def callback(grid_map):
    global latest_centroids, frame_id_global, length_x_global, length_y_global, resolution_global, position_x_global, position_y_global
    elevation_layer = grid_map.layers.index('elevation')
    frame_id_global = grid_map.info.header.frame_id
    resolution_global = float(grid_map.info.resolution)
    length_x_global = float(grid_map.info.length_x)
    length_y_global = float(grid_map.info.length_y)
    position_x_global = float(grid_map.info.pose.position.x)
    position_y_global = float(grid_map.info.pose.position.y)
    data_dim_x = int(np.ceil(length_x_global / resolution_global))
    data_dim_y = int(np.ceil(length_y_global / resolution_global))
    elevation_data = np.array(grid_map.data[elevation_layer].data, dtype=np.float32).reshape(data_dim_x, data_dim_y)
    
    # Extracting raw points before clustering
    raw_points = extract_raw_points(elevation_data)
    publish_raw_points(frame_id_global, resolution_global, length_x_global, length_y_global, raw_points)
    
    latest_centroids = define_rock_centroids(elevation_data)

def extract_raw_points(elevation_data):
    lower_elevation_threshold = -0.3
    upper_elevation_threshold = 0.2
    rock_mask = (elevation_data > lower_elevation_threshold) & (elevation_data < upper_elevation_threshold)
    
    x, y = np.where(rock_mask)
    points = np.column_stack((x, y))
    return points

def define_rock_centroids(elevation_data):
    points = extract_raw_points(elevation_data)
    
    clustering = DBSCAN(eps=4, min_samples=4).fit(points)
    labels = clustering.labels_
    
    unique_labels = set(labels)
    centroids = []
    for label in unique_labels:
        if label != -1:
            class_member_mask = (labels == label)
            class_points = points[class_member_mask]
            if 3 <= len(class_points) <= 25:
                centroid = np.mean(class_points, axis=0)
                centroids.append((int(centroid[0]), int(centroid[1]), 0))
    
    return centroids

def publish_raw_points(frame_id, resolution, length_x, length_y, points):
    raw_marker_array = MarkerArray()
    current_time = rospy.Time.now()
    offset_x = length_x / 2.0
    offset_y = length_y / 2.0
    
    for i, point in enumerate(points):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = current_time
        marker.type = Marker.SPHERE
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.pose.orientation.w = 1.0
        marker.lifetime = rospy.Duration(0.5)
        
        world_x = -point[1] * resolution + offset_y + position_x_global
        world_y = -point[0] * resolution + offset_x + position_y_global
        marker.pose.position.x = world_x
        marker.pose.position.y = world_y
        marker.pose.position.z = -0.1
        marker.id = i
        
        raw_marker_array.markers.append(marker)
    
    raw_marker_pub.publish(raw_marker_array)

def publish_markers(frame_id, resolution, length_x, length_y, centroids):
    global all_markers
    if all_markers is None:
        all_markers = MarkerArray()
    current_time = rospy.Time.now()
    current_marker_count = len(all_markers.markers)
    centroid_count = len(centroids)
    if centroid_count > current_marker_count:
        for i in range(current_marker_count, centroid_count):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.type = Marker.CUBE
            marker.scale.x = 0.6
            marker.scale.y = 0.6
            marker.scale.z = 0.6
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.pose.orientation.w = 1.0
            marker.lifetime = rospy.Duration(0.5)
            all_markers.markers.append(marker)
    elif centroid_count < current_marker_count:
        del all_markers.markers[centroid_count:]
    offset_x = length_x / 2.0
    offset_y = length_y / 2.0
    for i, centroid in enumerate(centroids):
        all_markers.markers[i].header.stamp = current_time
        world_x = -centroid[1] * resolution + offset_y + position_x_global
        world_y = -centroid[0] * resolution + offset_x + position_y_global
        all_markers.markers[i].pose.position.x = world_x
        all_markers.markers[i].pose.position.y = world_y
        all_markers.markers[i].pose.position.z = -0.1
        all_markers.markers[i].id = i
    marker_pub.publish(all_markers)

def timer_callback(event):
    if latest_centroids:
        publish_markers(frame_id_global, resolution_global, length_x_global, length_y_global, latest_centroids)

def listener():
    rospy.init_node('listener', anonymous=True)
    global marker_pub, raw_marker_pub
    marker_pub = rospy.Publisher("/rock_markers", MarkerArray, queue_size=1)
    raw_marker_pub = rospy.Publisher("/raw_rock_points", MarkerArray, queue_size=1)
    rospy.Subscriber("/elevation_mapping_large/elevation_map_raw", GridMap, callback)
    rospy.Timer(rospy.Duration(0.5), timer_callback, reset=True)
    rospy.spin()

if __name__ == '__main__':
    listener()