import threading

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from std_msgs.msg import String

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy
import xml.etree.ElementTree as ET


class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')

        # joint_states
        self.subscription = self.create_subscription(
            JointState,
            '~/joint_states',
            self.joint_states_cb,
            10)
        self.subscription  # prevent unused variable warning
        # Lock.
        self.lock = threading.Lock()

        # publish joint_trajectory
        self.publisher_joints = self.create_publisher(JointTrajectory, '~/joint_trajectory', 10)

        # URDF
        # Read the URDF file for the robot to ensure we have the correct joint names.
        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.urdf_sub = self.create_subscription(
            String,
            '/robot_description',
            self.urdf_callback,
            qos_profile  # Apply QoS profile
        )
        self.urdf_sub = None
        self.joints_from_urdf = {}


    def joint_states_cb(self, msg: JointState):
        if self.joints_from_urdf == {}:
            self.get_logger().error('Joint limits not set. Cannot publish trajectory.')
            return

        # as we get the message,
        with self.lock:
            self.joints_msg = msg

            new_msg = JointTrajectory()
            # process the message and convert to joint trajectory
            # TODO: convert this into a joint Trajectry and then publush.
            new_msg.joint_names = self.joints_msg.joint_names
            new_msg.header.stamp = self.get_clock().now().to_msg()

            point = JointTrajectoryPoint()
            joints_out = []
            for joint_name in new_msg.joint_names:
                self.get_logger().info(f"Processing joint: {joint_name}")
                min_limit, max_limit = self.joints_from_urdf[joint_name]
                self.get_logger().info(f"Joint {joint_name} limits: min={min_limit}, max={max_limit}")

                # Write square signal code
                # depending if 1 or 0, then we will publish the max or min limit.
                value = max_limit # TODO;

                joints_out.append(value)
            
            point.positions = joints_out
            point.velocities = [0.0] * len(joints_out)
            point.effort = [0.0] * len(joints_out)
            new_msg.points.append(point)
            self.publisher_joints.publish(new_msg)

    
    def urdf_callback(self, msg: String):
        """Extracts all joint names along with their min and max limits from the URDF."""
        urdf_str = msg.data
        try:
            root = ET.fromstring(urdf_str)
            for joint in root.findall('joint'):
                name = joint.get('name')
                limit = joint.find('limit')
                min_limit = limit.get('lower') if limit is not None else "N/A"
                max_limit = limit.get('upper') if limit is not None else "N/A"
                if min_limit == "N/A" or max_limit == "N/A":
                    self.get_logger().warn(f"Joint {name} has no limits. Will not add to the dictionary.")
                else:
                    self.joints_from_urdf[name] = (min_limit, max_limit)
        except ET.ParseError as e:
            self.get_logger().error(f"Failed to parse URDF: {e}")
        self.get_logger().info(f'Extracted Joints: {self.joints_from_urdf}')
        # assert len(self.joints_from_urdf.keys()) == self.action_size, \
        #     f"Number of joints in URDF ({len(self.joints_from_urdf.keys())}) does not match the action size ({self.action_size})."

            
            
"""
Sample /joint_states message:
header:
  stamp:
    sec: 1775186816
    nanosec: 435771226
  frame_id: ''
name:
- L_HAA
- L_HFE
- L_KFE
- L_ANKLE
- R_HAA
- R_HFE
- R_KFE
- R_ANKLE
position:
- -0.02804836930550095
- -0.4637710618221324
- 1.509127385445607
- -0.33687823945776424
- 0.13225430505260805
- -0.49269974489630186
- 0.9322264056305536
- -0.3889744035701289
velocity:
- -0.11271174718529363
- 0.2668163850602183
- -0.8909687765718356
- -0.17045010974534616
- -0.1318568398206744
- 0.32149634182859527
- -0.4247793277834653
- 0.15828551653038922
effort:
- 0.42357844461195526
- -0.29659630311354945
- 0.2179830828644782
- -0.0006444942958833896
- -4.897624149035736
- -1.2563663549775725
- -8.991236968494963
- 0.0031127215441812753
"""

"""
Sample /joint_trajectory message:
header:
  stamp:
    sec: 1775186867
    nanosec: 100098848
  frame_id: ''
joint_names:
- L_HAA
- L_HFE
- L_KFE
- L_ANKLE
- R_HAA
- R_HFE
- R_KFE
- R_ANKLE
points:
- positions:
  - 0.25
  - -1.071136736869812
  - 1.8434895472526551
  - -0.35
  - -0.25
  - -1.1152557671070098
  - 0.9741194341257214
  - -0.35
  velocities:
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  accelerations: []
  effort:
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  time_from_start:
    sec: 0
    nanosec: 0
"""

def main(args=None):
    try:
        rclpy.init(args=args)
        minimal_publisher = MinimalPublisher()

        rclpy.spin(minimal_publisher)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()