import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import TransformStamped
from motion_capture_tracking_interfaces.msg import NamedPoseArray

import numpy as np
import threading
from scipy.spatial.transform import Rotation as R

class TransformLookupNode(Node):
    def __init__(self):
        super().__init__('transform_lookup_node')

        # Initialize TF2 buffer and listener.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # params.
        self.declare_parameters(
            namespace='',
            parameters=[
                ('base_link_frame', 'base_link'),
                ('r_foot_frame', 'R_FOOT_CONTACT'),
                ('l_foot_frame', 'L_FOOT_CONTACT'),
                ('world_frame', 'world'),
                ('pose_base_link_mocap_name', 'biped'),
                ('dt', 0.01),
            ]
        )
        self.params = {
            'base_link_frame': self.get_parameter('base_link_frame').get_parameter_value().string_value,
            'r_foot_frame': self.get_parameter('r_foot_frame').get_parameter_value().string_value,
            'l_foot_frame': self.get_parameter('l_foot_frame').get_parameter_value().string_value,
            'world_frame': self.get_parameter('world_frame').get_parameter_value().string_value,
            'pose_base_link_mocap_name': self.get_parameter('pose_base_link_mocap_name').get_parameter_value().string_value,
            'dt': self.get_parameter('dt').get_parameter_value().double_value,
        }
        self.frames = [self.params['base_link_frame'], self.params['r_foot_frame'], self.params['l_foot_frame']]
        self.get_logger().info(f"Parameters: {self.params}")

        self.lock = threading.Lock()

        self.poses_vicon_msg = None
        self.sub_poses_vicon = self.create_subscription(NamedPoseArray, '/poses', self.vicon_callback, 10)

        self.points_base_link = []
        self.points_base_link_mocap = []

        self.timer = self.create_timer(self.params['dt'], self.run_ransac)

    def vicon_callback(self, msg: NamedPoseArray):
        with self.lock:
            # Ensure all frames are available.
            for frame in self.frames:
                try:
                    self.tf_buffer.lookup_transform(frame, self.params['world_frame'], rclpy.time.Time())
                except Exception as e:
                    self.get_logger().warn(f"Could not get transform: {e}")
                    return
                
            # Check the name exists in the /poses message.
            if not any([pose.name == self.params['pose_base_link_mocap_name'] for pose in msg.poses]):
                self.get_logger().warn(f"Could not find pose with name {self.params['pose_base_link_mocap_name']}")
                return

            # Get the pose of the base_link_mocap.
            self.poses_vicon_msg = msg
            for pose in self.poses_vicon_msg.poses:
                if pose.name == self.params['pose_base_link_mocap_name']:
                   pose_base_link_mocap = pose

            # Compute the rotation vector. For rotation, use a unit vector in the x-direction.
            v = np.array([1, 0, 0])
            q_base_link_mocap = [pose_base_link_mocap.pose.orientation.x,
                                            pose_base_link_mocap.pose.orientation.y,
                                            pose_base_link_mocap.pose.orientation.z,
                                            pose_base_link_mocap.pose.orientation.w]
            rot = R.from_quat(q_base_link_mocap)
            rotated_v_motion_capture = rot.apply(v)
            self.points_base_link_mocap.append(np.array([pose_base_link_mocap.pose.position.x,
                                                         pose_base_link_mocap.pose.position.y,
                                                         pose_base_link_mocap.pose.position.z,
                                                         rotated_v_motion_capture[0],
                                                         rotated_v_motion_capture[1],
                                                         rotated_v_motion_capture[2]]))
            
            # Get the base_link transform TF.
            base_link_trans, base_link_rot = self.lookup_transform(self.params['world_frame'], self.params['base_link_frame'])

            q_base_link = [base_link_rot.x, base_link_rot.y, base_link_rot.z, base_link_rot.w]
            rot = R.from_quat(q_base_link)
            rotated_v_base_link = rot.apply(v)

            self.points_base_link.append(np.array([base_link_trans.x,
                                                    base_link_trans.y,
                                                    base_link_trans.z, 
                                                    rotated_v_base_link[0],
                                                    rotated_v_base_link[1],
                                                    rotated_v_base_link[2]]))

    def lookup_transform(self, target_frame: str, source_frame: str):
        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time()
            )

            translation = transform.transform.translation
            rotation = transform.transform.rotation

            return translation, rotation

        except Exception as e:
            self.get_logger().warn(f"Could not get transform: {e}")
            return

    def ransac_translation(self, points_model, points_vicon, threshold=0.01, max_iterations=100):
        best_inliers = []
        best_translation = None
        N = 100

        if len(points_model) < N:
            raise ValueError("Not enough points to run RANSAC")

        for _ in range(max_iterations):
            indices = np.random.choice(len(points_model), N, replace=False)
            points_model_subset = points_model[indices]
            points_vicon_subset = points_vicon[indices]

            centroid_model = np.mean(points_model_subset, axis=0)
            centroid_vicon = np.mean(points_vicon_subset, axis=0)
            translation = centroid_vicon - centroid_model

            transformed_points = points_model + translation

            distances = np.linalg.norm(transformed_points - points_vicon, axis=1)
            inliers = np.where(distances < threshold)[0]

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_translation = translation

        if best_translation is not None and len(best_inliers) > 0:
            refined_translation = np.mean(points_vicon[best_inliers], axis=0) - np.mean(points_model[best_inliers], axis=0)
            return refined_translation, best_inliers
        else:
            raise ValueError("RANSAC failed to find a valid translation")

    def run_ransac(self):  
        if self.poses_vicon_msg is None:
            self.get_logger().warn(f"No VICON msg!")
            return

        assert np.array(self.points_base_link).shape == np.array(self.points_base_link_mocap).shape

        # Run RANSAC to find the best fit.
        try:
            refined_translation, inliers = self.ransac_translation(
                np.array(self.points_base_link),
                np.array(self.points_base_link_mocap)
            )
            pos = refined_translation[0:3]
            rot = refined_translation[3:]
            # Get rotation matrix.
            rot = R.from_rotvec(rot)
            self.get_logger().info(f"RANSAC translation: {refined_translation[0:3]}, rotation {rot.as_quat()}")
        except ValueError as e:
            self.get_logger().warn(f"RANSAC failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = TransformLookupNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()