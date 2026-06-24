#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point, PointStamped, Twist
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
import tf2_ros
import tf2_geometry_msgs
import numpy as np
import time
import math
import argparse
import stretch4_body.robot.robot_client as rc
from stretch4_body.utils.stretch_pose_models import RobotJoints
from stretch4_hybrid_marker_demos.joint_speeds import joint_speeds_dict
from tf2_geometry_msgs import do_transform_point
from control_msgs.msg import JointJog
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
import xml.etree.ElementTree as ET
from stretch4_urdf import get_urdf_from_robot_params

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
        self.gripper_name = RobotJoints.gripper.value 
        
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
        
        self.current_aperture = 0.15
        self.current_arm = 0.0
        self.sub_joint_states = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_states_callback,
            10
        )
        
        # Publisher for the target visualization
        self.pub_target = self.create_publisher(
            Marker,
            '/target_marker',
            10
        )
        
        # Publisher for joint velocity jogging
        self.pub_joint_vel = self.create_publisher(
            JointJog,
            'joint_vel',
            10
        )
        
        # Publisher for base velocity commands
        self.pub_cmd_vel = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )
        
        # Service client to set parameters on stretch_driver
        self.set_param_client = self.create_client(SetParameters, 'stretch_driver/set_parameters')
        if self.set_param_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info("Connected to 'stretch_driver/set_parameters' service.")
            future = self.switch_driver_mode('velocity')
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        else:
            self.get_logger().warn("Could not connect to 'stretch_driver/set_parameters' service. Mode-dependent features may fail.")
            
        # Robot Hardware Control
        self.robot = rc.RobotClient()
        if not self.robot.startup():
            self.get_logger().fatal("Failed to connect to Stretch hardware.")
            raise RuntimeError("Stretch hardware not found.")
            
        # 30Hz Control Loop Timer
        self.control_timer = self.create_timer(0.033, self.control_loop)
        
        # Parse retracted arm extension offset from URDF once at startup
        self.retracted_arm_x = self.get_retracted_arm_x()
        
        self.get_logger().info(f"Pursue Target Node started. Speed: {self.speed}, Trans: {self.enable_translation}, Rot: {self.enable_rotation}, Lift: {self.enable_lift}, Arm: {self.enable_arm}, Gripper: {self.enable_gripper}")
        self.get_logger().info("State: WAITING")

    def get_retracted_arm_x(self):
        
        try:
            root = ET.fromstring(get_urdf_from_robot_params())
            
            # Build parent-child and joint maps
            joints = {}
            for joint in root.findall('joint'):
                child_elem = joint.find('child')
                parent_elem = joint.find('parent')
                if child_elem is None or parent_elem is None:
                    continue
                child = child_elem.get('link')
                parent = parent_elem.get('link')
                
                # Get origin translation and rotation
                origin = joint.find('origin')
                xyz = [0.0, 0.0, 0.0]
                rpy = [0.0, 0.0, 0.0]
                if origin is not None:
                    if origin.get('xyz'):
                        xyz = [float(val) for val in origin.get('xyz').split()]
                    if origin.get('rpy'):
                        rpy = [float(val) for val in origin.get('rpy').split()]
                        
                joints[child] = {'parent': parent, 'xyz': xyz, 'rpy': rpy}
                
            # Helper for rotation matrix from Euler angles (roll, pitch, yaw)
            def rpy_to_matrix(r, p, y):
                Rx = np.array([[1.0, 0.0, 0.0],
                               [0.0, np.cos(r), -np.sin(r)],
                               [0.0, np.sin(r), np.cos(r)]])
                Ry = np.array([[np.cos(p), 0.0, np.sin(p)],
                               [0.0, 1.0, 0.0],
                               [-np.sin(p), 0.0, np.cos(p)]])
                Rz = np.array([[np.cos(y), -np.sin(y), 0.0],
                               [np.sin(y), np.cos(y), 0.0],
                               [0.0, 0.0, 1.0]])
                return Rz @ Ry @ Rx
                
            # Trace path from grasp center link to base_footprint depending on robot model
            path = []
            curr = 'grasp_center_link' if 'grasp_center_link' in joints else 'link_grasp_center'
            while curr in joints:
                joint_info = joints[curr]
                path.append(joint_info)
                curr = joint_info['parent']
                
            # Compute combined transformation matrix
            T = np.eye(4)
            for joint in reversed(path):
                r, p, y = joint['rpy']
                xyz = joint['xyz']
                R = rpy_to_matrix(r, p, y)
                T_step = np.eye(4)
                T_step[:3, :3] = R
                T_step[:3, 3] = xyz
                T = T @ T_step
                
            retracted_x = float(T[0, 3])
            self.get_logger().info(f"Parsed retracted grasp center X offset from URDF: {retracted_x:.4f}m")
            return retracted_x
        except Exception as e:
            self.get_logger().error(f"Failed to parse URDF for retracted arm X: {e}")
            return 0.18 # Safe fallback

    def joint_states_callback(self, msg):
        arm_pos = 0.0
        for name, pos in zip(msg.name, msg.position):
            if name == 'finger_left_joint':
                # Parallel gripper: physical position in meters is twice the finger displacement
                self.current_aperture = 2.0 * abs(pos)
            elif name == 'gripper_finger_left_joint':
                # Standard gripper (SG4): aperture = 2 * finger_length * sin(finger_rad)
                # Standard finger length is 0.0825m
                self.current_aperture = 2.0 * 0.0825 * math.sin(abs(pos))
            elif name in ['arm_l1_joint', 'arm_l2_joint', 'arm_l3_joint', 'arm_l4_joint']:
                arm_pos += pos
        self.current_arm = arm_pos

    def switch_driver_mode(self, mode_name):
        req = SetParameters.Request()
        param = Parameter()
        param.name = 'mode'
        param.value.type = ParameterType.PARAMETER_STRING
        param.value.string_value = mode_name
        req.parameters = [param]
        return self.set_param_client.call_async(req)

    def jog_gripper(self, displacement):
        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = [f"{self.gripper_name}_joint"]
        msg.velocities = [displacement / 300.0]
        self.pub_joint_vel.publish(msg)

    def send_base_velocity(self, v_x, v_y, omega):
        msg = Twist()
        msg.linear.x = float(v_x)
        msg.linear.y = float(v_y)
        msg.angular.z = float(omega)
        self.pub_cmd_vel.publish(msg)

    def publish_joint_velocities(self, joint_velocities):
        """
        Publishes a JointJog message with the specified joint velocities.
        joint_velocities is a dict, e.g. {'lift_joint': v_lift, 'arm_joint': v_arm}
        """
        if not joint_velocities:
            return
        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = list(joint_velocities.keys())
        msg.velocities = list(joint_velocities.values())
        self.pub_joint_vel.publish(msg)

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

        # Lookup grasp center TF to dynamically get gripper position depending on robot model
        base_to_grasp_x = None
        base_to_grasp_y = None
        base_to_grasp_z = None
        for grasp_frame in ["grasp_center_link", "grasp_frame_link"]: 
            try:
                t_base_footprint_to_grasp = self.tf_buffer.lookup_transform('base_footprint', grasp_frame, rclpy.time.Time())
                base_to_grasp_x = t_base_footprint_to_grasp.transform.translation.x
                base_to_grasp_y = t_base_footprint_to_grasp.transform.translation.y
                base_to_grasp_z = t_base_footprint_to_grasp.transform.translation.z
                break
            except Exception:
                continue
                
        if base_to_grasp_x is None:
            self.get_logger().warn("Failed to lookup grasp center TF (tried 'grasp_center_link' and 'link_grasp_center')")
            return
                
        # Check target position availability for states requiring target tracking
        if self.state in ["FIXATED", "GRASPING"]:
            if not hasattr(self, 'target_pos_base') or self.target_pos_base is None:
                return
            base_to_target_x, base_to_target_y, base_to_target_z = self.target_pos_base
        else:
            base_to_target_x = 0.0
            base_to_target_y = 0.0
            base_to_target_z = 0.0

        # Translational error to bring the target into the gripper's geometric setpoint.
        grasp_to_target_x = base_to_target_x - base_to_grasp_x 
        grasp_to_target_y = base_to_target_y - base_to_grasp_y
        grasp_to_target_z = base_to_target_z - base_to_grasp_z

        # Rotational error to align the gripper's forward line of action with the target.
        grasp_to_target_theta = math.atan2(grasp_to_target_x, grasp_to_target_y)

        if self.state == "FIXATED" and self.target_features is not None:
            # target_features is currently in the frame_id of the MarkerArray (usually /world or base_footprint)
            # We must map it precisely to base_footprint to compute error.
            # However, we only have the point. Let's just lookup the TF and transform it natively.
            
            e_x = grasp_to_target_x - 0.20
            e_y = grasp_to_target_y
            e_z = grasp_to_target_z
            e_theta = math.atan2(e_x, e_y)

            # Proportional Gains mapped to requested speed
            Kp = { 'slow': 0.1, 'default': 0.12, 'fast': 0.2, 'max': 0.5 }.get(self.speed, 0.12)
            Kw = { 'slow': 0.5, 'default': 0.75, 'fast': 1.0, 'max': 2.0 }.get(self.speed, 0.75)
            
            # Base Velocity Control
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
                
            self.send_base_velocity(v_x, v_y, omega)
            
            # Lift, Arm & Gripper Velocity Control
            joint_vels = {}
            if self.enable_lift:
                # Track target height (Z-axis offset) with the lift joint
                joint_vels['lift_joint'] = float(np.clip(Kp * e_z, -self.lift_v, self.lift_v))
            if self.enable_arm:
                # Keep arm retracted at a safe position (0.01m) during target alignment/fixation
                e_arm = 0.01 - self.current_arm
                joint_vels['arm_joint'] = float(np.clip(3.0 * e_arm, -self.arm_v, self.arm_v))
            if self.enable_gripper:
                # Keep the gripper open while fixating on target to prepare for grasp
                joint_vels[f"{self.gripper_name}_joint"] = self.gripper_pct / 300.0
            
            if joint_vels:
                self.publish_joint_velocities(joint_vels)
            
        elif self.state == "GRASPING" and self.target_features is not None:
            # Base must not move
            self.send_base_velocity(0.0, 0.0, 0.0)

            joint_vels = {}
            # Align lift with target Z height during grasp
            if self.enable_lift:
                joint_vels['lift_joint'] = float(np.clip(Kp * grasp_to_target_z, -self.lift_v, self.lift_v))

            # Get current aperture dynamically from joint states topic
            current_aperture = self.current_aperture
            target_aperture = 0.03
            if self.enable_gripper:
                # Goal: Close the gripper to target aperture (0.03m) to secure the object
                # Closed-loop tracking for constant aperture with wider deadband to prevent oscillation
                if current_aperture > (target_aperture + 0.015):
                    joint_vels[f"{self.gripper_name}_joint"] = -self.gripper_pct / 300.0
                elif current_aperture < (target_aperture - 0.015):
                    joint_vels[f"{self.gripper_name}_joint"] = self.gripper_pct / 300.0
                    
            # Reach out or retract based on grasp success
            if self.enable_arm:
                # If within 0.05m of target aperture, consider grasp successful and retract
                if abs(current_aperture - target_aperture) <= 0.05:
                    e_arm = 0.01 - self.current_arm
                else:
                    # Extend the arm until there is no x error relative to target along grasp center's X-axis
                    e_arm = grasp_to_target_x
                joint_vels['arm_joint'] = float(np.clip(3.0 * e_arm, -self.arm_v, self.arm_v))
                
            if joint_vels:
                self.publish_joint_velocities(joint_vels)
    
        elif self.state == "WAITING":
            # Just ensure we are stopped
            self.send_base_velocity(0.0, 0.0, 0.0)
            
            joint_vels = {}
            # Reposition Arm
            if self.enable_arm:
                # Goal: Keep the arm retracted (0.01m) while waiting for target detection
                e_arm = 0.01 - self.current_arm
                joint_vels['arm_joint'] = float(np.clip(3.0 * e_arm, -self.arm_v, self.arm_v))
            
            # Gripper Proportional Tracker (Lightly close when Waiting)
            if self.enable_gripper:
                # Goal: Gently close/maintain gripper aperture (0.03m) while waiting to reduce footprint
                current_aperture = self.current_aperture
                target_aperture = 0.03
                
                # Maintain the closed-loop constant aperture with very gentle steps and wide deadband
                if current_aperture > (target_aperture + 0.015):
                    joint_vels[f"{self.gripper_name}_joint"] = -5.0 / 300.0
                elif current_aperture < (target_aperture - 0.015):
                    joint_vels[f"{self.gripper_name}_joint"] = 5.0 / 300.0
                    
            if joint_vels:
                self.publish_joint_velocities(joint_vels)
            
    def destroy_node(self):
        # Switch mode back to navigation
        if hasattr(self, 'set_param_client') and self.set_param_client.service_is_ready():
            self.get_logger().info("Resetting stretch_driver mode to navigation...")
            req = SetParameters.Request()
            param = Parameter()
            param.name = 'mode'
            param.value.type = ParameterType.PARAMETER_STRING
            param.value.string_value = 'navigation'
            req.parameters = [param]
            future = self.set_param_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
            
        # Stop the base
        self.send_base_velocity(0.0, 0.0, 0.0)
        super().destroy_node()
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
