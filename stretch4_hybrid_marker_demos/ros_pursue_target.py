#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from builtin_interfaces.msg import Duration
import tf2_ros
import tf2_geometry_msgs
import numpy as np
import time
import math
import argparse
import stretch4_body.robot.robot_client as rc

from stretch4_hybrid_marker_demos.joint_speeds import joint_speeds_dict

class PursueTargetNode(Node):
    def __init__(self, speed='default', enable_translation=True, enable_rotation=True, enable_lift=True, enable_arm=True, enable_gripper=True):
        super().__init__('pursue_target_node')
        
        # Base Control Configuration
        self.speed = speed
        self.enable_translation = enable_translation
        self.enable_rotation = enable_rotation
        self.enable_lift = enable_lift
        self.enable_arm = enable_arm
        self.enable_gripper = enable_gripper
        
        # Load independent joint dynamics configurations from joint_speeds.py
        if self.speed not in joint_speeds_dict['lift'] or self.speed not in joint_speeds_dict['gripper'] or self.speed not in joint_speeds_dict['arm']:
            self.get_logger().error(f"Configured speed '{self.speed}' not found in joint_speeds_dict. Defaulting to 'default'.")
            self.speed = 'default'
            
        self.lift_v = joint_speeds_dict['lift'][self.speed]['vel_m']
        self.lift_a = joint_speeds_dict['lift'][self.speed]['accel_m']
        
        self.arm_v = joint_speeds_dict['arm'][self.speed]['vel_m']
        self.arm_a = joint_speeds_dict['arm'][self.speed]['accel_m']
        
        self.gripper_v = joint_speeds_dict['gripper'][self.speed]['vel']
        self.gripper_a = joint_speeds_dict['gripper'][self.speed]['accel']
        self.gripper_pct = {
            'slow': 10.0,
            'default': 30.0,
            'fast': 60.0,
            'max': 60.0
        }.get(self.speed, 15.0)
        self.wait_time_sec = 2.0
        self.wait_count_min = 20
        self.agent_count_min = 5
        self.agent_color_thresh = 0.5 # 128 / 255.0 on Red channel
        
        # Bounding box in base_footprint
        self.bb_x_min = 0.4
        self.bb_x_max = 1.5
        self.bb_y_min = -0.5
        self.bb_y_max = 0.5
        self.bb_z_min = 0.3
        self.bb_z_max = 1.5
        
        # State tracking
        # WAITING -> FIXATED
        self.state = "WAITING"
        self.target_id = None
        self.target_features = None  # (x,y,z) in world/published frame
        
        # Candidate tracking for WAITING state
        # Dict mapping track_id -> {'first_seen': float, 'count': int, 'agent_count': int}
        self.candidates = {}
        
        # TF Buffer to transform marker coordinates into base_footprint 
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Subscriptions
        self.sub = self.create_subscription(
            MarkerArray,
            '/lidar_tracked_clusters',
            self.marker_callback,
            10
        )
        
        # Publisher for the target visualization
        self.pub_target = self.create_publisher(
            Marker,
            '/target_marker',
            10
        )
        
        # Robot Hardware Control
        self.robot = rc.RobotClient()
        if not self.robot.startup():
            self.get_logger().fatal("Failed to connect to Stretch hardware.")
            raise RuntimeError("Stretch hardware not found.")
            
        # 30Hz Control Loop Timer
        self.control_timer = self.create_timer(0.033, self.control_loop)
        
        self.get_logger().info(f"Pursue Target Node started. Speed: {self.speed}, Trans: {self.enable_translation}, Rot: {self.enable_rotation}, Lift: {self.enable_lift}, Arm: {self.enable_arm}, Gripper: {self.enable_gripper}")
        self.get_logger().info("State: WAITING")

    def transform_point_to_base(self, pt_msg, from_frame, stamp):
        """
        Transforms a geometry_msgs/Point from `from_frame` to `base_footprint`.
        Returns (x, y, z) tuple or None if transform fails.
        """
        if from_frame == 'base_footprint':
            return (pt_msg.x, pt_msg.y, pt_msg.z)
            
        try:
            # We want to know where the point is relative to the robot's base *now* or at the stamp time
            # Since MarkerArray uses the exact stamp of the TF tree, this should succeed.
            t = self.tf_buffer.lookup_transform('base_footprint', from_frame, stamp, rclpy.duration.Duration(seconds=0.1))
            
            # Use tf2_geometry_msgs to transform the point
            from tf2_geometry_msgs import do_transform_point
            from geometry_msgs.msg import PointStamped
            
            ps = PointStamped()
            ps.header.frame_id = from_frame
            ps.header.stamp = stamp
            ps.point = pt_msg
            
            ps_base = do_transform_point(ps, t)
            return (ps_base.point.x, ps_base.point.y, ps_base.point.z)
            
        except Exception as e:
            self.get_logger().debug(f"TF Error transforming point: {e}")
            return None

    def marker_callback(self, msg):
        now = time.time()
        
        # Since MarkerArray has multiple markers per track (sphere, text, arrow),
        # we only care about extracting the cluster features once per ID.
        # The SPHERE markers encode the categorization score as RGB and the position.
        active_ids = set()
        
        for marker in msg.markers:
            if marker.action == Marker.DELETEALL:
                continue
                
            if marker.type != Marker.SPHERE:
                continue
                
            track_id = marker.id
            active_ids.add(track_id)
            
            frame_id = marker.header.frame_id
            stamp = marker.header.stamp
            
            # Read properties
            pt = marker.pose.position
            red_score = marker.color.r # Agent Categorization score
            
            if self.state in ["FIXATED", "GRASPING"]:
                if track_id == self.target_id:
                    self.target_features = pt
                    
                    # Update base relative position for the control loop
                    pt_base = self.transform_point_to_base(pt, frame_id, stamp)
                    if pt_base is not None:
                        self.target_pos_base = pt_base
                        
                    self.publish_target_marker(header=marker.header, pt=pt)
            
            elif self.state == "WAITING":
                # Check Spatial Bounding Box conditionally
                pt_base = self.transform_point_to_base(pt, frame_id, stamp)
                if pt_base is None:
                    continue # TF not ready, skip evaluating this frame
                    
                self.target_pos_base = pt_base # Cache for the control loop
                x, y, z = pt_base
                
                in_box = (self.bb_x_min <= x <= self.bb_x_max and
                          self.bb_y_min <= y <= self.bb_y_max and
                          self.bb_z_min <= z <= self.bb_z_max)
                          
                if in_box:
                    if track_id not in self.candidates:
                        self.candidates[track_id] = {
                            'first_seen': now,
                            'count': 0,
                            'agent_count': 0
                        }
                        
                    metric = self.candidates[track_id]
                    metric['count'] += 1
                    
                    if red_score >= self.agent_color_thresh:
                        metric['agent_count'] += 1
                        
                    # Evaluate Lock Conditions
                    elapsed = now - metric['first_seen']
                    if (elapsed >= self.wait_time_sec and 
                        metric['count'] >= self.wait_count_min and 
                        metric['agent_count'] >= self.agent_count_min):
                        
                        self.target_id = track_id
                        self.state = "FIXATED"
                        self.get_logger().info(f"Target Locked! ID: {track_id}. Transitioning to FIXATED.")
                        
                        # Stop evaluating other candidates
                        self.candidates.clear()
                        self.target_features = pt
                        self.publish_target_marker(header=marker.header, pt=pt)
                        break # Skip rest of markers
                else:
                    # If it steps out of the box, reset its counters so it has to stay in continuously
                    self.candidates.pop(track_id, None)

        # Post-process missing tracks and state transitions
        if self.state in ["FIXATED", "GRASPING"]:
            if self.target_id not in active_ids:
                self.get_logger().warn(f"Target Lost! ID: {self.target_id} disappeared. Transitioning to WAITING.")
                self.state = "WAITING"
                self.target_id = None
                self.target_features = None
                # RViz2 will automatically clear the marker due to its short lifetime
                
        elif self.state == "WAITING":
            # Clean up dead candidates
            dead_ids = list(set(self.candidates.keys()) - active_ids)
            for d in dead_ids:
                self.candidates.pop(d, None)

    def publish_target_marker(self, header, pt):
        marker = Marker()
        marker.header = header
        marker.ns = "pursuit_target"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.lifetime = Duration(sec=0, nanosec=200000000) # 0.2 seconds lifetime
        
        marker.pose.position = pt
        marker.pose.orientation.w = 1.0
        
        # Make a large obvious bounding box (halved width)
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.15
        
        # Bright Yellow / Orange
        marker.color.r = 1.0
        marker.color.g = 0.8
        marker.color.b = 0.0
        marker.color.a = 0.5
        
        self.pub_target.publish(marker)

    def control_loop(self):
        """
        Runs at 30Hz to command the omnibase to pursue the target object.
        Goal is to keep the target at the gripper's offset (x=0.5m, y=-0.095m relative to base).
        """
        # Pull latest hardware status before control calculations
        self.robot.pull_status()
        
        if self.state == "FIXATED" and self.target_features is not None:
            # target_features is currently in the frame_id of the MarkerArray (usually /world or base_footprint)
            # We must map it precisely to base_footprint to compute error.
            # However, we only have the point. Let's just lookup the TF and transform it natively.
            
            # Since target_features is just the geometric point stored during marker_callback,
            # we need its frame and stamp. Instead of saving those explicitly, we can just save the 
            # transformed pt_base directly in marker_callback. Let's just calculate the error directly
            # from a cached base_footprint coordinate.
            pass # See modified marker_callback for self.target_pos_base 
            
            if not hasattr(self, 'target_pos_base') or self.target_pos_base is None:
                return
                
            tx, ty, tz = self.target_pos_base
            
            # Simple Kinematic Model for the Gripper Pursuit
            # The gripper is located to the right of the base center line
            gripper_offset_y = -0.095
            
            # We want the target to be 0.5m forward relative to the base, and centered on the gripper
            target_distance_x = 0.6
            
            # Map the target coordinates into the gripper's frame of reference
            tx_gripper = tx
            ty_gripper = ty - gripper_offset_y
            
            # Translational error to bring the target into the gripper's geometric setpoint
            e_x = tx_gripper - target_distance_x
            e_y = ty_gripper - 0.0
            
            # Rotational error to aim the gripper's forward line of action at the target.
            # By tracking the offset angle, the robot's center line points 0.095m to the left of the target,
            # which perfectly aligns the right-mounted gripper.
            e_theta = math.atan2(ty_gripper, tx_gripper) 
            
            # Proportional Gains mapped to requested speed
            Kp = { 'slow': 0.1, 'default': 0.12, 'fast': 0.2, 'max': 0.5 }.get(self.speed, 0.12)
            Kw = { 'slow': 0.5, 'default': 0.75, 'fast': 1.0, 'max': 2.0 }.get(self.speed, 0.75)
            
            v_x = Kp * e_x if self.enable_translation else 0.0
            v_y = Kp * e_y if self.enable_translation else 0.0
            omega = Kw * e_theta if self.enable_rotation else 0.0
            
            # Deadband evaluation
            dist_error = math.sqrt(e_x**2 + e_y**2)
            if dist_error < 0.01:
                v_x = 0.0
                v_y = 0.0
                
            # State Transition: Check if we are close enough to Grasp
            if dist_error < 0.02:
                self.get_logger().info(f"Target in range (dist: {dist_error:.3f}m)! Transitioning to GRASPING.")
                self.state = "GRASPING"
                return # Base stops natively next loop
                
            if abs(e_theta) < math.radians(2.0):
                omega = 0.0
                
            self.robot.base.set_velocity(v_x, v_y, omega)
            
            # Lift Proportional Position Tracker
            if self.enable_lift:
                target_lift = tz - 0.14
                # Clip rigidly to safe bounds
                cmd_lift = float(np.clip(target_lift, 0.2, 1.1))
                
                # Check deadband for the lift to prevent constant micro-adjustments
                # Need to read the current lift position; we can query status if it's updated, 
                # but move_to with a small delta usually handles this natively in the firmware safely.
                self.robot.lift.move_to(cmd_lift, v_m=self.lift_v, a_m=self.lift_a)
                
            # Arm Keep-out Zone (Retracted)
            if self.enable_arm:
                self.robot.arm.move_to(0.01, v_m=self.arm_v, a_m=self.arm_a)
                
            # Gripper Proportional Tracker (Open when Fixated)
            if self.enable_gripper:
                self.robot.end_of_arm.move_by('stretch_gripper', self.gripper_pct, self.gripper_v, self.gripper_a)
                
            self.robot.push_command()
            
        elif self.state == "GRASPING" and self.target_features is not None:
            # Base must not move
            self.robot.base.set_velocity(0.0, 0.0, 0.0)
            
            # Still track lift height if possible to not smash it against edges
            if hasattr(self, 'target_pos_base') and self.target_pos_base is not None and self.enable_lift:
                tz = self.target_pos_base[2]
                target_lift = tz - 0.14
                cmd_lift = float(np.clip(target_lift, 0.2, 1.1))
                self.robot.lift.move_to(cmd_lift, v_m=self.lift_v, a_m=self.lift_a)
                
            # Read gripper status unconditionally for both arm and gripper logic
            gripper_status = self.robot.end_of_arm.status.get('stretch_gripper', {})
            current_aperture = float(gripper_status.get('gripper_conversion', {}).get('aperture_m', 0.15))
            target_aperture = 0.03
            print('current aperture: ', current_aperture)
            if self.enable_gripper:
                # Closed-loop tracking for constant aperture with wider deadband to prevent oscillation
                if current_aperture > (target_aperture + 0.015):
                    self.robot.end_of_arm.move_by('stretch_gripper', -self.gripper_pct, self.gripper_v, self.gripper_a)
                elif current_aperture < (target_aperture - 0.015):
                    self.robot.end_of_arm.move_by('stretch_gripper', self.gripper_pct, self.gripper_v, self.gripper_a)
                    
            # Reach out or retract based on grasp success
            if self.enable_arm:
                # If within 0.05m of target aperture, consider grasp successful and retract
                if abs(current_aperture - target_aperture) <= 0.05:
                    self.robot.arm.move_to(0.01, v_m=self.arm_v, a_m=self.arm_a)
                else:
                    self.robot.arm.move_to(0.12, v_m=self.arm_v, a_m=self.arm_a)

            self.robot.push_command()
            
        elif self.state == "WAITING":
            # Just ensure we are stopped
            self.robot.base.set_velocity(0.0, 0.0, 0.0)
            
            # Reposition Arm
            if self.enable_arm:
                self.robot.arm.move_to(0.01, v_m=self.arm_v, a_m=self.arm_a)
            
            # Gripper Proportional Tracker (Lightly close when Waiting)
            if self.enable_gripper:
                gripper_status = self.robot.end_of_arm.status.get('stretch_gripper', {})
                current_aperture = float(gripper_status.get('gripper_conversion', {}).get('aperture_m', 0.15))
                target_aperture = 0.03
                
                # Maintain the closed-loop constant aperture with very gentle steps and wide deadband
                if current_aperture > (target_aperture + 0.015):
                    self.robot.end_of_arm.move_by('stretch_gripper', -5.0, self.gripper_v, self.gripper_a)
                elif current_aperture < (target_aperture - 0.015):
                    self.robot.end_of_arm.move_by('stretch_gripper', 5.0, self.gripper_v, self.gripper_a)
                    
            self.robot.push_command()
            
    def destroy_node(self):
        super().destroy_node()
        self.robot.base.set_velocity(0.0, 0.0, 0.0)
        self.robot.push_command()
        self.robot.stop()

def main():
    parser = argparse.ArgumentParser(description='Omnibase Target Pursuit for Stretch.')
    parser.add_argument('--speed', type=str, default='default', choices=['slow', 'default', 'fast', 'max'],
                        help='Speed profile for pursuing the target.')
    parser.add_argument('--enable_translation', action='store_true', default=True,
                        help='Enable standard XY base translation towards the target.')
    parser.add_argument('--disable_translation', action='store_false', dest='enable_translation',
                        help='Disable standard XY base translation towards the target.')
    parser.add_argument('--enable_rotation', action='store_true', default=True,
                        help='Enable base rotation to point at the target.')
    parser.add_argument('--disable_rotation', action='store_false', dest='enable_rotation',
                        help='Disable base rotation.')
    parser.add_argument('--enable_lift', action='store_true', default=True,
                        help='Enable lift to track target height.')
    parser.add_argument('--disable_lift', action='store_false', dest='enable_lift',
                        help='Disable lift height tracking.')
    parser.add_argument('--enable_gripper', action='store_true', default=True,
                        help='Enable gripper to dynamically open and close.')
    parser.add_argument('--disable_gripper', action='store_false', dest='enable_gripper',
                        help='Disable gripper dynamics.')
    parser.add_argument('--enable_arm', action='store_true', default=True,
                        help='Enable arm extension.')
    parser.add_argument('--disable_arm', action='store_false', dest='enable_arm',
                        help='Disable arm actions.')
                        
    args, ros_args = parser.parse_known_args()
    
    import sys
    new_argv = [sys.argv[0]] + ros_args
    rclpy.init(args=new_argv)
    
    node = PursueTargetNode(
        speed=args.speed,
        enable_translation=args.enable_translation,
        enable_rotation=args.enable_rotation,
        enable_lift=args.enable_lift,
        enable_arm=args.enable_arm,
        enable_gripper=args.enable_gripper
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
