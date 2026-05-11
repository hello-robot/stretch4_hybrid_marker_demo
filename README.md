# stretch4_hybrid_marker_demo

This repository provides demonstrations of the use of hybrid markers with the Stretch 4 mobile manipulator from Hello Robot Inc. Hybrid markers combine LiDAR-reflective material with a visible light ArUco marker to support efficient and robust autonomy.

This package is structured as a standard ROS 2 Python package. 

## Installation

You have two options for installing this package: using the standard ROS 2 `colcon` build system or installing it directly as a regular Python package via `pip`.

### Option 1: Standard ROS 2 Install (colcon)

To install this package in a ROS 2 workspace, clone it into your workspace's `src` directory and build it using `colcon`:

```bash
# Assuming your workspace is ~/ros2_ws
cd ~/ros2_ws/src
git clone https://github.com/hello-robot/stretch4_hybrid_marker_demos.git
cd ~/ros2_ws
colcon build --packages-select stretch4_hybrid_marker_demos
source install/setup.bash
```

### Option 2: Python Package Install (pip)

If you prefer to avoid the complexities of a full ROS 2 workspace, you can install the package directly into your current Python environment (or virtual environment) using `pip`. The package is configured as a standard `setuptools` Python package:

```bash
git clone https://github.com/hello-robot/stretch4_hybrid_marker_demos.git
cd stretch4_hybrid_marker_demos
pip install -e .
```
*(Make sure you have sourced your base ROS 2 installation first so `rclpy` and other dependencies are available).*

## Usage

To run the full hybrid marker pursuit demo, you must start several components in separate terminals. First, ensure the foundational robot drivers and protocols are running:

**Terminal 1: Launch Stretch Driver**
```bash
ros2 launch stretch_core stretch_driver.launch.py
```

**Terminal 2: Launch the Hesai LiDAR**
```bash
ros2 launch stretch_core dual_hesai.launch.py
```

**Terminal 3: Launch the Zenoh pub/sub/query protocol**
```bash
ros2 run rmw_zenoh_cpp rmw_zenohd
```

**Terminal 4: Start the RViz2 visualization**
```bash
cd ~/repos/stretch4_hybrid_marker_demos
rviz2 --display-config ./rviz/pursue_target.rviz
```

> [!NOTE]
> **Important Calibration Step:** For best performance, the calibrated transform between `base_footprint` and `base_link` should be used. However, the latest Stretch 4 URDF has a `base_footprint` frame with a nominal transform that can conflict with this broadcasted static transform (a known issue to be fixed in the future). If you wish to use the calibrated transform, run the following in a separate terminal from the calibration repository:
> ```bash
> cd ~/repos/stretch_dual_lidar_calibration
> python3 ./stretch_dual_lidar_calibration/ros_broadcast_calibration.py
> # Expected Output: [INFO] [...] [calibration_broadcaster]: Broadcasted static transform base_link -> base_footprint
> ```

### Running the Demo Nodes

Depending on how you installed the package, use one of the following methods in **Terminal 5** and **Terminal 6** to start the tracker and pursuit nodes.

#### Method A: Using Standard ROS 2 Install (colcon)

**Terminal 5: Start the high-intensity LiDAR object tracker**
```bash
ros2 run stretch4_hybrid_marker_demos ros_track_object
```

**Terminal 6: Start the hybrid marker cube pursuit demo**
```bash
ros2 run stretch4_hybrid_marker_demos ros_pursue_target
```

#### Method B: Using Simple Python Install (pip)

**Terminal 5: Start the high-intensity LiDAR object tracker**
```bash
cd ~/repos/stretch4_hybrid_marker_demos
python3 ./stretch4_hybrid_marker_demos/ros_track_object.py
```

**Terminal 6: Start the hybrid marker cube pursuit demo**
```bash
cd ~/repos/stretch4_hybrid_marker_demos
python3 ./stretch4_hybrid_marker_demos/ros_pursue_target.py
```

### Command Line Arguments

You can customize the pursuit behavior using several command-line flags:

- `--speed {slow,default,fast,max}`: Sets the dynamic speed profile for the robot's base, lift, arm, and gripper (default: `default`).
- `--disable_translation`: Prevents the base from driving linearly toward the target (X/Y translation).
- `--disable_rotation`: Prevents the base from rotating to face the target.
- `--disable_lift`: Prevents the lift from dynamically tracking the target's height.
- `--disable_arm`: Disables arm extension/retraction behavior.
- `--disable_gripper`: Disables dynamic opening/closing of the gripper.

**Example with custom arguments:**
```bash
ros2 run stretch4_hybrid_marker_demos ros_pursue_target --speed fast --disable_arm
```

## Technical Details: How the Demo Works

The `ros_pursue_target` demo is implemented as a ROS 2 node that interacts with the robot hardware using `stretch_body_ii` and subscribes to perception topics from the hybrid marker tracking pipeline.

### 1. State Machine
The robot operates via a simple finite state machine consisting of three states:
- **WAITING**: The robot evaluates the incoming clusters and waits for a robust track to fulfill lock-on conditions.
- **FIXATED**: The robot actively commands its joints (base, lift, arm, gripper) to pursue the locked target and align its gripper with the target's position.
- **GRASPING**: Initiated when the robot is sufficiently close to the target. The base stops, and the arm and gripper attempt to engage.

### 2. Target Acquisition & Filtering
The node subscribes to `/lidar_tracked_clusters` (a `visualization_msgs/MarkerArray`). It filters for tracks that appear as `SPHERE` markers, using the `color.r` channel to extract the "agent categorization score" generated by the upstream perception stack. 

To prevent false positives, a target candidate must satisfy three conditions:
1. It must remain within a predefined 3D spatial bounding box relative to the `base_footprint` (in front of the robot).
2. It must be continuously tracked for at least 2.0 seconds (`wait_time_sec`).
3. It must have at least 5 highly confident "agent" classifications (`agent_count_min`).

### 3. Control Loop & Hardware Interface
A 30Hz ROS 2 timer (`self.control_timer`) dictates the hardware control loop.
- **Transformations**: The target's coordinates are continuously transformed into the `base_footprint` frame using `tf2_ros` so that the robot can accurately calculate positional errors regardless of the camera's pose.
- **Kinematic Pursuit**: A proportional controller reduces the distance and angular error between the target and the gripper. The system accounts for the gripper's right-side offset (`gripper_offset_y = -0.095m`) and aims for a set distance in front of the base (`target_distance_x = 0.6m`).
- **Command Aggregation**: Joint commands for the base, lift, arm, and gripper are aggregated and sent to the firmware simultaneously using `self.robot.push_command()` to ensure synchronized motion.

### 4. Grasp Evaluation
When the distance error drops below 0.02 meters, the state transitions to **GRASPING**. The base holds position while the end-of-arm attempts a grasp. If the gripper's aperture matches the target aperture, the grasp is considered successful, and the arm retracts.
