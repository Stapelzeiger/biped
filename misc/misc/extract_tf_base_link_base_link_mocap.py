import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import TransformStamped
from motion_capture_tracking_interfaces.msg import NamedPoseArray

import numpy as np
import threading
from scipy.spatial.transform import Rotation as R
import pandas as pd

def apply_rotation_to_vector(q):
    '''
    Apply a quaternion rotation to a unit vector [1, 0, 0].

    q: quaternion [x, y, z, w]
    Returns: Rotated vector as a numpy array.
    '''
    print(q)
    v = np.array([1, 0, 0])
    rot = R.from_quat([q.x, q.y, q.z, q.w])
    rotated_v = rot.apply(v)
    return rotated_v

class RansacNode(Node):
    def __init__(self):
        super().__init__('transform_lookup_node')

        # Initialize TF2 buffer and listener.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Params.
        self.declare_parameters(
            namespace='',
            parameters=[
                ('base_link_frame', 'base_link'),
                ('r_foot_frame', 'R_FOOT_CONTACT'),
                ('l_foot_frame', 'L_FOOT_CONTACT'),
                ('world_frame', 'world'),
                ('pose_base_link_mocap_name', 'biped'),
                ('pose_right_foot_mocap_name', 'right_foot'),
                ('pose_left_foot_mocap_name', 'left_foot'),
                ('dt', 0.01),
                ('save_to_csv', True)
            ]
        )
        self.params = {
            'base_link_frame': self.get_parameter('base_link_frame').get_parameter_value().string_value,
            'r_foot_frame': self.get_parameter('r_foot_frame').get_parameter_value().string_value,
            'l_foot_frame': self.get_parameter('l_foot_frame').get_parameter_value().string_value,
            'world_frame': self.get_parameter('world_frame').get_parameter_value().string_value,
            'pose_base_link_mocap_name': self.get_parameter('pose_base_link_mocap_name').get_parameter_value().string_value,
            'pose_right_foot_mocap_name': self.get_parameter('pose_right_foot_mocap_name').get_parameter_value().string_value,
            'pose_left_foot_mocap_name': self.get_parameter('pose_left_foot_mocap_name').get_parameter_value().string_value,
            'dt': self.get_parameter('dt').get_parameter_value().double_value,
            'save_to_csv': self.get_parameter('save_to_csv').get_parameter_value().bool_value
        }
        self.frames = [self.params['base_link_frame'], self.params['r_foot_frame'], self.params['l_foot_frame']]
        self.get_logger().info(f"Parameters: {self.params}")

        self.list_refined_translation = []

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
            if not any([pose.name == self.params['pose_right_foot_mocap_name'] for pose in msg.poses]):
                self.get_logger().warn(f"Could not find pose with name {self.params['pose_right_foot_mocap_name']}")
                return
            if not any([pose.name == self.params['pose_left_foot_mocap_name'] for pose in msg.poses]):
                self.get_logger().warn(f"Could not find pose with name {self.params['pose_left_foot_mocap_name']}")
                return

            # Get the pose from mocap.
            self.poses_vicon_msg = msg
            for pose in self.poses_vicon_msg.poses:
                if pose.name == self.params['pose_base_link_mocap_name']:
                   pose_base_link_mocap = pose
                elif pose.name == self.params['pose_right_foot_mocap_name']:
                    pose_right_foot_mocap = pose
                elif pose.name == self.params['pose_left_foot_mocap_name']:
                    pose_left_foot_mocap = pose

            # Compute the rotation vector. For rotation, use a unit vector in the x-direction.
            rotated_v_base_link_mocap = apply_rotation_to_vector(pose_base_link_mocap.pose.orientation)
            rotated_v_right_foot_mocap = apply_rotation_to_vector(pose_right_foot_mocap.pose.orientation)
            rotated_v_left_foot_mocap = apply_rotation_to_vector(pose_left_foot_mocap.pose.orientation)
            
            row_mocap = [pose_base_link_mocap.pose.position.x,
                        pose_base_link_mocap.pose.position.y,
                        pose_base_link_mocap.pose.position.z,
                        rotated_v_base_link_mocap[0],
                        rotated_v_base_link_mocap[1],
                        rotated_v_base_link_mocap[2],
                        pose_right_foot_mocap.pose.position.x,
                        pose_right_foot_mocap.pose.position.y,
                        pose_right_foot_mocap.pose.position.z,
                        rotated_v_right_foot_mocap[0],
                        rotated_v_right_foot_mocap[1],
                        rotated_v_right_foot_mocap[2],
                        pose_left_foot_mocap.pose.position.x,
                        pose_left_foot_mocap.pose.position.y,
                        pose_left_foot_mocap.pose.position.z,
                        rotated_v_left_foot_mocap[0],
                        rotated_v_left_foot_mocap[1],
                        rotated_v_left_foot_mocap[2]]
            self.points_base_link_mocap.append(np.array(row_mocap))

            # Get the TFs.
            base_link_trans, base_link_rot = self.lookup_transform(self.params['world_frame'], self.params['base_link_frame'])
            right_foot_trans, right_foot_rot = self.lookup_transform(self.params['world_frame'], self.params['r_foot_frame'])
            left_foot_trans, left_foot_rot = self.lookup_transform(self.params['world_frame'], self.params['l_foot_frame'])

            rotated_v_base_link = apply_rotation_to_vector(base_link_rot)
            rotated_v_right_foot = apply_rotation_to_vector(right_foot_rot)
            rotated_v_left_foot = apply_rotation_to_vector(left_foot_rot)

            row_tfs = [base_link_trans.x,
                        base_link_trans.y,
                        base_link_trans.z,
                        rotated_v_base_link[0],
                        rotated_v_base_link[1],
                        rotated_v_base_link[2],
                        right_foot_trans.x,
                        right_foot_trans.y,
                        right_foot_trans.z,
                        rotated_v_right_foot[0],
                        rotated_v_right_foot[1],
                        rotated_v_right_foot[2],
                        left_foot_trans.x,
                        left_foot_trans.y,
                        left_foot_trans.z,
                        rotated_v_left_foot[0],
                        rotated_v_left_foot[1],
                        rotated_v_left_foot[2]]
            self.points_base_link.append(np.array(row_tfs))

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

    def ransac_translation(self, points_model, points_vicon, threshold=0.1, max_iterations=100):
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
            # self.get_logger().info(f"Refined translation: {refined_translation}")

            if self.params['save_to_csv']:
                # Save to csv.
                # self.get_logger().info("Saving to CSV")
                self.list_refined_translation.append(refined_translation)
                df = pd.DataFrame(self.list_refined_translation)
                df.to_csv('/home/sorina/Documents/code/biped_hardware/ros2_ws/src/biped/misc/misc/ransac_data/refined_translation.csv', index=False)
        except ValueError as e:
            self.get_logger().warn(f"RANSAC failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = RansacNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()