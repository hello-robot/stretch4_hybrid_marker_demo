#!/usr/bin/env python3
import sys
import argparse
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs_py import point_cloud2
from message_filters import Subscriber, ApproximateTimeSynchronizer
import tf2_ros
from geometry_msgs.msg import TransformStamped, Point
import numpy as np

from stretch_dual_lidar_calibration.dual_lidar_calibration import DualLidarCalibration
from stretch_dual_lidar_calibration.lidar_utils import LidarProcessor
from stretch4_hybrid_marker_demos.point_clustering import cluster_points_kdtree, cluster_points_iterative_sphere
from stretch4_hybrid_marker_demos.cluster_tracking import AVAILABLE_TRACKERS

class TrackObjectNode(Node):
    def __init__(self, clustering_method='kdtree', 
                 cluster_connectivity_distance=0.02, 
                 cluster_min_points=5, 
                 cluster_max_radius=0.15, 
                 cluster_min_density=0.0,
                 tracking_method='alpha_beta_hungarian',
                 tracker_max_match_distance=1.5,
                 tracker_max_staleness=3,
                 tracker_alpha=0.8,
                 tracker_beta=0.4,
                 cat_v_static_thresh=0.05,
                 cat_d_robot_max=0.5,
                 cat_use_ransac=True,
                 cat_ransac_threshold=0.1,
                 cat_world_tf_timeout=0.5):
        super().__init__('track_object_node')
        
        # ROS Parameters (for topics only)
        self.declare_parameter('left_lidar_topic', '/lidar_points_left')
        self.declare_parameter('right_lidar_topic', '/lidar_points_right')
        
        self.left_topic = self.get_parameter('left_lidar_topic').value
        self.right_topic = self.get_parameter('right_lidar_topic').value
        
        # Argparse Parameters
        self.clustering_method = clustering_method
        self.cluster_connectivity_distance = cluster_connectivity_distance
        self.cluster_min_points = cluster_min_points
        self.cluster_max_radius = cluster_max_radius
        self.cluster_min_density = cluster_min_density
        
        # Tracking Parameters
        self.tracking_method = tracking_method
        
        # Initialize Tracker
        if self.tracking_method not in AVAILABLE_TRACKERS:
            error_msg = f"Unknown tracking method: '{self.tracking_method}'. Available: {list(AVAILABLE_TRACKERS.keys())}"
            self.get_logger().fatal(error_msg)
            raise ValueError(error_msg)
            
        TrackerClass = AVAILABLE_TRACKERS[self.tracking_method]
        self.tracker = TrackerClass(
            max_match_distance=tracker_max_match_distance, 
            max_staleness=tracker_max_staleness,
            alpha=tracker_alpha,
            beta=tracker_beta
        )
        
        self.last_frame_time = None
        
        # Categorization Parameters
        self.cat_v_static_thresh = cat_v_static_thresh # m/s
        self.cat_d_robot_max = cat_d_robot_max # meters radius from base frame origin
        self.cat_use_ransac = cat_use_ransac
        self.cat_ransac_threshold = cat_ransac_threshold
        self.cat_world_tf_timeout = cat_world_tf_timeout
        self.world_tf_active = False # Tracks if we're currently using the world tf or falling back
        
        # Load calibration
        self.calibration = DualLidarCalibration()
        if not self.calibration.load():
            self.get_logger().error("Could not load calibration file!")
            
        self.lidar_processor = LidarProcessor(self.calibration)
        
        # TF Buffer for lookups
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Subscriptions
        self.left_sub = Subscriber(self, PointCloud2, self.left_topic)
        self.right_sub = Subscriber(self, PointCloud2, self.right_topic)
        
        self.ts = ApproximateTimeSynchronizer(
            [self.right_sub, self.left_sub],
            queue_size=10,
            slop=0.05,
            allow_headerless=False
        )
        self.ts.registerCallback(self.cloud_callback)
        
        # Publisher
        self.pub_tracked_object = self.create_publisher(PointCloud2, 'lidar_tracked_object', 10)
        self.pub_tracked_clusters = self.create_publisher(MarkerArray, 'lidar_tracked_clusters', 10)
        
        self.get_logger().info("Track Object Node started.")

    def filter_high_intensity(self, msg):
        try:
            # Read x, y, z, intensity
            data = point_cloud2.read_points_numpy(msg, field_names=['x', 'y', 'z', 'intensity'], skip_nans=True)
            if len(data) > 0:
                mask = data[:, 3] >= 255.0  # intensities >= 255 (retroreflective)
                filtered_data = data[mask]
            else:
                filtered_data = np.zeros((0, 4), dtype=np.float32)
        except Exception as e:
            self.get_logger().warn(f"Failed to read points with intensity: {e}")
            filtered_data = np.zeros((0, 4), dtype=np.float32)

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        
        return point_cloud2.create_cloud(msg.header, fields, filtered_data)

    def cluster_points(self, points):
        """
        Clusters sparse high-intensity 3D points using the selected method.
        Filters clusters based on configurable constraints.
        Returns a list of dicts containing centroid, radius, point array, and indices.
        """
        if len(points) == 0:
            return []
            
        method = self.clustering_method
        
        if method == 'kdtree':
            return cluster_points_kdtree(points[:, :3], 
                                         self.cluster_connectivity_distance, 
                                         self.cluster_min_points, 
                                         self.cluster_max_radius)
            
        elif method == 'iterative_sphere':
            return cluster_points_iterative_sphere(points[:, :3], 
                                                   self.cluster_max_radius, 
                                                   self.cluster_min_points, 
                                                   self.cluster_min_density)
            
        else:
            error_msg = f"Unknown clustering method: '{method}'. Valid options are 'kdtree' or 'iterative_sphere'."
            self.get_logger().error(error_msg)
            raise ValueError(error_msg)

    def estimate_planar_transform_ransac(self, positions, velocities, max_iterations=50, threshold=0.1):
        """
        Fits a 2D Rigid Body Motion model (v_x, v_y) = (V_x - w * y, V_y + w * x)
        to a set of cluster positions and velocities in the base_footprint frame using RANSAC.
        Returns the best fitting model (Vx, Vy, w) and the indices of the inlier points.
        """
        n_points = len(positions)
        if n_points < 2:
            return None, np.array([], dtype=int)
            
        best_inlier_count = 0
        best_model = None
        best_inliers = np.array([], dtype=int)
        
        for _ in range(max_iterations):
            sample_idx = np.random.choice(n_points, 2, replace=False)
            p_sample = positions[sample_idx]
            v_sample = velocities[sample_idx]
            
            A_sample = np.array([
                [1, 0, -p_sample[0, 1]],
                [0, 1,  p_sample[0, 0]],
                [1, 0, -p_sample[1, 1]],
                [0, 1,  p_sample[1, 0]]
            ])
            b_sample = np.array([
                v_sample[0, 0], v_sample[0, 1],
                v_sample[1, 0], v_sample[1, 1]
            ])
            
            try:
                model, _, _, _ = np.linalg.lstsq(A_sample, b_sample, rcond=None)
            except np.linalg.LinAlgError:
                continue
                
            A_all = np.zeros((2 * n_points, 3))
            A_all[0::2, 0] = 1
            A_all[0::2, 2] = -positions[:, 1]
            A_all[1::2, 1] = 1
            A_all[1::2, 2] = positions[:, 0]
            
            b_pred = A_all @ model
            
            errors_x = b_pred[0::2] - velocities[:, 0]
            errors_y = b_pred[1::2] - velocities[:, 1]
            
            # include v_z error heavily since planar motion assumes v_z = 0
            total_errors = np.sqrt(errors_x**2 + errors_y**2 + velocities[:, 2]**2)
            
            inliers = np.where(total_errors < threshold)[0]
            
            if len(inliers) > best_inlier_count:
                best_inlier_count = len(inliers)
                best_model = model
                best_inliers = inliers
                
        # Refit on all inliers
        if len(best_inliers) >= 2:
            p_in = positions[best_inliers]
            v_in = velocities[best_inliers]
            A_in = np.zeros((2 * len(best_inliers), 3))
            A_in[0::2, 0] = 1
            A_in[0::2, 2] = -p_in[:, 1]
            A_in[1::2, 1] = 1
            A_in[1::2, 2] = p_in[:, 0]
            
            b_in = np.zeros(2 * len(best_inliers))
            b_in[0::2] = v_in[:, 0]
            b_in[1::2] = v_in[:, 1]
            
            best_model, _, _, _ = np.linalg.lstsq(A_in, b_in, rcond=None)
                
        return best_model, best_inliers

    def categorize_tracks(self, active_tracks, stamp, frame_id, world_to_base=None):
        """
        Computes likelihoods for Robot, Environment, and Agent for each active track
        and updates the smoothed probability scores via the TrackedCluster. 
        """
        if not active_tracks:
            return
            
        n_tracks = len(active_tracks)
        positions = np.array([t.features[0:3] for t in active_tracks])
        velocities = np.array([t.velocity for t in active_tracks])
        
        # Calculate distances to the robot. If in base_footprint, robot is at (0,0).
        # If in world, robot is at world_to_base translation.
        if frame_id == 'world' and world_to_base is not None:
            robot_pos = np.array([world_to_base.transform.translation.x, 
                                  world_to_base.transform.translation.y, 
                                  world_to_base.transform.translation.z])
            distances_to_robot = np.linalg.norm(positions[:, 0:2] - robot_pos[0:2], axis=1)
        else:
            distances_to_robot = np.linalg.norm(positions[:, 0:2], axis=1) # xy distance
            
        speed = np.linalg.norm(velocities, axis=1)
        
        # Heuristics for all methods
        l_robot = np.zeros(n_tracks)
        l_env = np.zeros(n_tracks)
        l_agent = np.zeros(n_tracks)
        
        # 1. Robot Likelihood
        # High if near origin and low velocity relative to base frame
        for i in range(n_tracks):
            if distances_to_robot[i] < self.cat_d_robot_max:
                if frame_id == 'world':
                    # If in world frame, the robot's world velocity might be non-zero.
                    # It's safest to just say if it's consistently very close to the center, it's the robot.
                    l_robot[i] = 1.0
                else:
                    if speed[i] < self.cat_v_static_thresh:
                        l_robot[i] = 1.0 # Very confident it's the robot body
                    elif speed[i] < self.cat_v_static_thresh * 2:
                        l_robot[i] = 0.5 # Margin cases
                
        # 2. Environment Likelihood
        if frame_id == 'world':
            # We are natively tracking in the world frame.
            # Environment points are stationary in the world natively.
            for i in range(n_tracks):
                if l_robot[i] > 0.8:
                    continue
                if speed[i] < self.cat_v_static_thresh:
                    l_env[i] = 1.0
                elif speed[i] < self.cat_v_static_thresh * 2:
                    l_env[i] = 0.5
        else:
            # We are in base_footprint, use RANSAC (or planar heuristics)
            env_model, env_inliers = None, []
            if self.cat_use_ransac and n_tracks >= 2:
                env_model, env_inliers = self.estimate_planar_transform_ransac(
                    positions, velocities, threshold=self.cat_ransac_threshold)
                    
            for i in range(n_tracks):
                if l_robot[i] > 0.8:
                    continue # Ignore robot parts for env/agent
                    
                if self.cat_use_ransac and env_model is not None:
                    # Is it an inlier to the rigid body world transform?
                    if i in env_inliers:
                        l_env[i] = 1.0
                    else:
                        # Calculate continuous error for soft score
                        p = positions[i]
                        v = velocities[i]
                        v_pred_x = env_model[0] - env_model[2] * p[1]
                        v_pred_y = env_model[1] + env_model[2] * p[0]
                        err = np.sqrt((v_pred_x - v[0])**2 + (v_pred_y - v[1])**2 + v[2]**2)
                        
                        if err < self.cat_ransac_threshold * 2:
                            l_env[i] = 0.5
                else:
                    # Fallback: Simple purely planar heuristic
                    # World shouldn't be zooming up and down relative to the robot
                    if abs(velocities[i, 2]) < 0.1 and distances_to_robot[i] > self.cat_d_robot_max:
                        l_env[i] = 0.8
                    
        # 3. Agent Likelihood (Active independent motion)
        for i in range(n_tracks):
            # If it's definitely not the robot and definitely not the environment, it's an agent
            base_agent = 1.0 - max(l_robot[i], l_env[i])
            
            # Give explicit agent bonus if the z-velocity is high (lifting)
            if abs(velocities[i, 2]) > 0.15:
                base_agent = max(base_agent, 1.0)
                
            l_agent[i] = base_agent
            
        # Update trackers with smoothed scores
        for i, track in enumerate(active_tracks):
            track.update_scores(l_robot[i], l_env[i], l_agent[i], smooth_alpha=0.3)

    def create_track_markers(self, active_tracks, header):
        marker_array = MarkerArray()
        
        # Note: We append to the same MarkerArray as the clusters to keep RViz clean,
        # but put them in a different namespace
        
        for track in active_tracks:
            # 1. Text Marker for Track ID
            text_marker = Marker()
            text_marker.header = header
            text_marker.ns = "track_ids"
            text_marker.id = track.track_id # Unique persistent ID
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            # Position slightly above the cluster
            text_marker.pose.position.x = float(track.features[0])
            text_marker.pose.position.y = float(track.features[1])
            text_marker.pose.position.z = float(track.features[2] + track.features[3] + 0.1)
            
            text_marker.scale.z = 0.1 # Text height
            
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            
            # Show ID and Velocity magnitude
            vel_mag = np.linalg.norm(track.velocity)
            text_marker.text = f"ID:{track.track_id} V:{vel_mag:.2f}m/s"
            
            marker_array.markers.append(text_marker)
            
            # 2. Arrow Marker for Velocity Vector
            if vel_mag > 0.05: # Only draw meaningful vectors
                arrow_marker = Marker()
                arrow_marker.header = header
                arrow_marker.ns = "track_velocities"
                arrow_marker.id = track.track_id
                arrow_marker.type = Marker.ARROW
                arrow_marker.action = Marker.ADD
                
                # Start point
                p1 = Point()
                p1.x, p1.y, p1.z = float(track.features[0]), float(track.features[1]), float(track.features[2])
                
                # End point (scale velocity for visibility, e.g. 1 second prediction)
                p2 = Point()
                p2.x = float(track.features[0] + track.velocity[0])
                p2.y = float(track.features[1] + track.velocity[1])
                p2.z = float(track.features[2] + track.velocity[2])
                
                arrow_marker.points = [p1, p2]
                
                arrow_marker.scale.x = 0.02 # Shaft diameter
                arrow_marker.scale.y = 0.04 # Head diameter
                
                arrow_marker.color.r = 1.0
                arrow_marker.color.g = 0.2
                arrow_marker.color.b = 0.2
                arrow_marker.color.a = 0.8
                
                marker_array.markers.append(arrow_marker)
                
        return marker_array

    def transform_points_kdl(self, points, transform):
        """
        Transforms an Nx3 numpy array of points using a geometry_msgs/TransformStamped
        """
        qx = transform.transform.rotation.x
        qy = transform.transform.rotation.y
        qz = transform.transform.rotation.z
        qw = transform.transform.rotation.w
        
        # 3x3 rotation matrix
        R = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])
        
        T = np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z
        ])
        
        xyz = points[:, :3]
        transformed_xyz = (R @ xyz.T).T + T
        
        if points.shape[1] > 3:
            out = np.copy(points)
            out[:, :3] = transformed_xyz
            return out
        else:
            return transformed_xyz

    def cloud_callback(self, right_msg, left_msg):
        # Calculate dt
        current_time = self.get_clock().now()
        dt = 0.1 # default 10Hz if first frame
        if self.last_frame_time is not None:
            # rclpy duration to nanoseconds, then to seconds
            dt = (current_time - self.last_frame_time).nanoseconds / 1e9
        self.last_frame_time = current_time
        
        # Safety bound dt
        if dt <= 0 or dt > 1.0:
            dt = 0.1
            
        # 1. Filter out LiDAR points with intensities < 255
        left_filtered = self.filter_high_intensity(left_msg)
        right_filtered = self.filter_high_intensity(right_msg)
        
        # 2. Unify filtered clouds using LidarProcessor
        points_fp = self.lidar_processor.unify_clouds(left_filtered, right_filtered, self.tf_buffer)
        
        if points_fp is None or len(points_fp) == 0:
            return
            
        stamp = left_msg.header.stamp
        frame_id = 'base_footprint'
        world_to_base = None
        
        # 3. Apply World TF if available. This significantly improves tracking
        # and allows trivial environment categorization.
        # Use Time() to get the LATEST available transform, since pose_estimate
        # naturally lags behind the raw high-frequency LiDAR scans.
        if self.tf_buffer.can_transform('world', 'base_footprint', rclpy.time.Time()):
            try:
                world_to_base = self.tf_buffer.lookup_transform('world', 'base_footprint', rclpy.time.Time())
                
                # Sync our outgoing stamp exactly to the TF tree to prevent RViz extrapolation errors
                tf_stamp = world_to_base.header.stamp
                tf_time = rclpy.time.Time.from_msg(tf_stamp)
                msg_time = rclpy.time.Time.from_msg(stamp)
                
                # Check for stale TF (e.g., ros_estimate_pose.py crashed/stopped)
                age_ns = (msg_time - tf_time).nanoseconds
                if age_ns > self.cat_world_tf_timeout * 1e9:
                    if self.world_tf_active:
                        self.get_logger().warn(f"World TF is too old ({(age_ns/1e9):.2f}s). Falling back to base_footprint.")
                        self.world_tf_active = False
                    world_to_base = None
                else:
                    if not self.world_tf_active:
                        self.get_logger().info("World TF is active and current. Tracking in /world frame.")
                        self.world_tf_active = True
                        
                    stamp = tf_stamp
                    
                    # Apply Transform to all points
                    points_fp = self.transform_points_kdl(points_fp, world_to_base)
                    frame_id = 'world'
            except Exception as e:
                # If we were previously active and now it failed, log it
                if self.world_tf_active:
                    self.get_logger().warn(f"World TF lookup failed: {e}. Falling back to base_footprint.")
                    self.world_tf_active = False
                
        # 4. Cluster the high intensity 3D points
        clusters = self.cluster_points(points_fp)
        
        # 5. Temporal Tracking
        active_tracks = self.tracker.update(clusters, dt)
        
        # 6. Categorize tracked clusters (Robot/Env/Agent)
        self.categorize_tracks(active_tracks, stamp, frame_id, world_to_base)
        
        # 7. Extract valid points back into a unified cloud based on ACTIVE TRACKS
        if len(active_tracks) > 0:
            valid_indices = []
            for track in active_tracks:
                # We saved the raw cluster points into the track during the update step
                if hasattr(track, 'latest_cluster') and track.latest_cluster:
                    valid_indices.extend(track.latest_cluster['indices'])
            points_fp = points_fp[valid_indices] if len(valid_indices) > 0 else np.zeros((0, points_fp.shape[1]), dtype=np.float32)
        else:
            points_fp = np.zeros((0, points_fp.shape[1]), dtype=np.float32)

        # 6. Publish the resulting aligned, unified point cloud of high intensity points
        header = self.get_header(stamp, frame_id)
        msg_pub = LidarProcessor.create_cloud(header, points_fp)
        self.pub_tracked_object.publish(msg_pub)
        
        # 7. Publish spherical approximations and IDs of the clusters
        
        # Hack to clear old markers
        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        # Add basic spheres
        for i, track in enumerate(active_tracks):
            marker = Marker()
            marker.header = header
            marker.ns = "tracked_clusters"
            marker.id = track.track_id # Persistent ID!
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.pose.position.x = float(track.features[0])
            marker.pose.position.y = float(track.features[1])
            marker.pose.position.z = float(track.features[2])
            marker.pose.orientation.w = 1.0
            
            # Features[3] is radius
            diameter = max(0.05, 2.0 * float(track.features[3]))
            marker.scale.x = diameter
            marker.scale.y = diameter
            marker.scale.z = diameter
            
            # Map Categorization Scores to RGB: Red=Agent, Green=Env, Blue=Robot
            r = track.scores['agent']
            g = track.scores['env']
            b = track.scores['robot']
            
            # Normalize so the strongest component hits 1.0 (maximum brightness 255)
            max_c = max(r, g, b)
            if max_c > 0:
                r /= max_c
                g /= max_c
                b /= max_c
                
            marker.color.r = float(r)
            marker.color.g = float(g)
            marker.color.b = float(b)
            marker.color.a = 0.8 # Slightly more opaque to see colors clearly 
            
            marker_array.markers.append(marker)
            
        # Append Tracking Data (IDs, Velocity Vectors)
        tracking_markers = self.create_track_markers(active_tracks, header)
        marker_array.markers.extend(tracking_markers.markers)
        
        self.pub_tracked_clusters.publish(marker_array)

    def get_header(self, stamp=None, frame_id='base_footprint'):
        h = TransformStamped().header
        if stamp is None:
            h.stamp = self.get_clock().now().to_msg()
        else:
            h.stamp = stamp
        h.frame_id = frame_id
        return h

def main():
    parser = argparse.ArgumentParser(description='Object tracking ROS 2 node using Dual LiDARs.')
    
    # Python CLI Parameters
    parser.add_argument('--clustering_method', type=str, default='kdtree',
                        choices=['kdtree', 'iterative_sphere'],
                        help='Clustering algorithm to use.')
    parser.add_argument('--cluster_connectivity_distance', type=float, default=0.02,
                        help='Max distance between points in a cluster (for kdtree).')
    parser.add_argument('--cluster_min_points', type=int, default=5,
                        help='Minimum points to form a valid cluster.')
    parser.add_argument('--cluster_max_radius', type=float, default=0.15,
                        help='Maximum radius of the bounding sphere.')
    parser.add_argument('--cluster_min_density', type=float, default=0.0,
                        help='Minimum points per cubic meter (for iterative_sphere).')
    parser.add_argument('--tracking_method', type=str, default='alpha_beta_hungarian',
                        choices=list(AVAILABLE_TRACKERS.keys()),
                        help='Temporal tracking algorithm to use.')
    parser.add_argument('--tracker_max_match_distance', type=float, default=1.5,
                        help='Max feature distance to associate a track to a detection.')
    parser.add_argument('--tracker_max_staleness', type=int, default=3,
                        help='Number of frames a track can be unseen before dropping.')
    parser.add_argument('--tracker_alpha', type=float, default=0.8,
                        help='Alpha-Beta filter pos smoothing (higher = trusts new measurements more).')
    parser.add_argument('--tracker_beta', type=float, default=0.4,
                        help='Alpha-Beta filter vel smoothing (higher = updates velocity faster).')
                        
    # Categorization Parameters
    parser.add_argument('--cat_v_static_thresh', type=float, default=0.05,
                        help='Max velocity to be considered a static Robot body part (m/s).')
    parser.add_argument('--cat_d_robot_max', type=float, default=0.5,
                        help='Max distance from center to consider Robot body part.')
    parser.add_argument('--cat_use_ransac', type=bool, default=True,
                        help='Enable RANSAC robust planar motion tracking for Environment.')
    parser.add_argument('--cat_ransac_threshold', type=float, default=0.1,
                        help='Velocity error threshold (m/s) for Environment RANSAC inliers.')
    parser.add_argument('--cat_world_tf_timeout', type=float, default=0.5,
                        help='Maximum age (seconds) of the /world TF before falling back to RANSAC.')
    
    # Parse Python parameters but allow ROS 2 specific args to pass through unhindered
    args, ros_args = parser.parse_known_args()
    
    # We must construct a new argv because rclpy.init expects sys.argv format
    # which includes the python script name at index 0.
    new_argv = [sys.argv[0]] + ros_args
        
    rclpy.init(args=new_argv)
    
    # Pass parsed argparse rules as direct kwargs to our node initializer
    node = TrackObjectNode(
        clustering_method=args.clustering_method,
        cluster_connectivity_distance=args.cluster_connectivity_distance,
        cluster_min_points=args.cluster_min_points,
        cluster_max_radius=args.cluster_max_radius,
        cluster_min_density=args.cluster_min_density,
        tracking_method=args.tracking_method,
        tracker_max_match_distance=args.tracker_max_match_distance,
        tracker_max_staleness=args.tracker_max_staleness,
        tracker_alpha=args.tracker_alpha,
        tracker_beta=args.tracker_beta,
        cat_v_static_thresh=args.cat_v_static_thresh,
        cat_d_robot_max=args.cat_d_robot_max,
        cat_use_ransac=args.cat_use_ransac,
        cat_ransac_threshold=args.cat_ransac_threshold,
        cat_world_tf_timeout=args.cat_world_tf_timeout
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
