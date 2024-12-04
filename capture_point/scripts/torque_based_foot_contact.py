#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from std_msgs.msg import Bool
import numpy as np
from biped_bringup.msg import StampedBool

class FootContactDetector(Node):
    def __init__(self):
        super().__init__('foot_contact_detector')

        self.L_FOOT_in_contact = False
        self.R_FOOT_in_contact = False

        list_joints = ["L_KFE", "L_HFE", "L_HAA", "R_KFE", "R_HFE", "R_HAA"]
        
        self.joints_eff_hist = {joint: [] for joint in list_joints}        
        self.joints_effort = {joint: None for joint in list_joints}
        self.joints_effort_filtered = {joint: None for joint in list_joints}

        self.des_joints_effort = {joint: None for joint in list_joints}

        self.delta_joints_eff = {joint: None for joint in list_joints}
        self.delta_joints_eff_hist = {joint: [] for joint in list_joints}
        self.delta_joints_eff_filtered = {joint: None for joint in list_joints}

        # Filter params for the joint effort signal.
        self.filter_window_size = 1

        # Sim
        self.threshold_eff_KFE = 4.0 # Nm
        self.threashold_eff_HAA = 5.0 # Nm
        
        # Hardware
        # self.threshold_eff_KFE = 5.0 # Nm
        # self.threashold_eff_HFE = 8.0 # Nm

        self.msg_traj = None

        self.contact_publisher_left = self.create_publisher(StampedBool, '~/left_foot_contact', 10)
        self.contact_publisher_right = self.create_publisher(StampedBool, '~/right_foot_contact', 10)
        self.sub_joint_states = self.create_subscription(JointState, '/joint_states', self.joint_effort_callback, 10)
        self.sub_joint_traj = self.create_subscription(JointTrajectory, '/joint_trajectory', self.joint_traj_callback, 10)
        self.pub_filtered_torque = self.create_publisher(JointState, '~/filtered_joint_states', 10)
        self.pub_delta_torque = self.create_publisher(JointState, '~/delta_joint_states', 10)

    def low_pass_filter(self, lst, value):
        lst.append(value)
        if len(lst) > self.filter_window_size:
            lst.pop(0)
        return np.mean(lst), lst
    
    def joint_traj_callback(self, msg):
        self.msg_traj = msg
        
    def joint_effort_callback(self, msg):
        now = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9
        
        if not msg.effort:
            self.get_logger().warn("No effort data in the message")
            return
        
        time_msg = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if time_msg - now > 0.1:
            # print msg.header.stamp.sec - self.get_clock().now().to_msg().sec
            self.get_logger().warn("msg.header.stamp.sec - self.get_clock().now().to_msg().sec" + str(time_msg))
            self.get_logger().warn("Joint State message is too old")
            return
        if msg.name == []:
            self.get_logger().warn("No joint names in the message")
            return
        
        if self.msg_traj is None:
            self.get_logger().warn("Trajectory message is not available")
            return
        if self.msg_traj is not None:
            time_msg_traj = self.msg_traj.header.stamp.sec + self.msg_traj.header.stamp.nanosec * 1e-9
            if abs(time_msg_traj - now) > 0.1:
                self.get_logger().warn("Trajectory message is too old by" + str(time_msg_traj - now))
                return
        
        if abs(time_msg - time_msg_traj) > 0.1:
            self.get_logger().warn("Joint state and trajectory message are not synchronized, diff is " + str(abs(time_msg - time_msg_traj)))
            return

        # Joint effort.
        for i, joint_name in enumerate(msg.name):
            for key, _ in self.joints_effort.items():
                if joint_name == key:
                    self.joints_effort[key] = msg.effort[i]

        # Filter joint effort.
        for key, _ in self.joints_effort_filtered.items():
            if self.joints_effort[key] is not None:
                self.joints_effort_filtered[key], self.joints_eff_hist[key] = self.low_pass_filter(self.joints_eff_hist[key], self.joints_effort[key])

        # Desired joint effort.
        for i, joint_name in enumerate(self.msg_traj.joint_names):
            for key, _ in self.des_joints_effort.items():
                if joint_name == key:
                    self.des_joints_effort[key] = self.msg_traj.points[0].effort[i]

        # Delta joint effort.
        for key, _ in self.delta_joints_eff.items():
            if self.joints_effort_filtered[key] is not None:
                self.delta_joints_eff[key] = abs(self.joints_effort[key] - self.des_joints_effort[key])

        # Filter delta_joints_eff.
        for key, _ in self.delta_joints_eff_filtered.items():
            if self.delta_joints_eff[key] is not None:
                self.delta_joints_eff_filtered[key], self.delta_joints_eff_hist[key] = self.low_pass_filter(self.delta_joints_eff_hist[key], self.delta_joints_eff[key])

        # Edge detection.
        if self.delta_joints_eff_filtered["L_KFE"] > self.threshold_eff_KFE or self.delta_joints_eff_filtered["L_HAA"] > self.threashold_eff_HAA:
            self.L_FOOT_in_contact = True
        else:
            self.L_FOOT_in_contact = False
        
        if self.delta_joints_eff_filtered["R_KFE"] > self.threshold_eff_KFE or self.delta_joints_eff_filtered["R_HAA"] > self.threashold_eff_HAA:
            self.R_FOOT_in_contact = True
        else:
            self.R_FOOT_in_contact = False

        # ROS2 messages.
        msg_filtered = JointState()
        msg_filtered = msg
        for i, joint_name in enumerate(msg_filtered.name):
            for key, _ in self.joints_effort_filtered.items():
                if joint_name == key:
                    msg_filtered.effort[i] = self.joints_effort_filtered[key]
        self.pub_filtered_torque.publish(msg_filtered)

        msg_left_contact = StampedBool()
        msg_left_contact.header.stamp.sec = self.get_clock().now().to_msg().sec
        msg_left_contact.header.stamp.nanosec = self.get_clock().now().to_msg().nanosec
        msg_left_contact.data = self.L_FOOT_in_contact
        self.contact_publisher_left.publish(msg_left_contact)

        msg_right_contact = StampedBool()
        msg_right_contact.header.stamp.sec = self.get_clock().now().to_msg().sec
        msg_right_contact.header.stamp.nanosec = self.get_clock().now().to_msg().nanosec
        msg_right_contact.data = self.R_FOOT_in_contact
        self.contact_publisher_right.publish(msg_right_contact)

        msg_delta_joint_states = JointState()
        msg_delta_joint_states.header.stamp.sec = self.get_clock().now().to_msg().sec
        msg_delta_joint_states.header.stamp.nanosec = self.get_clock().now().to_msg().nanosec
        msg_delta_joint_states.name = self.delta_joints_eff_filtered.keys()
        msg_delta_joint_states.effort = [value for value in self.delta_joints_eff_filtered.values()]
        self.pub_delta_torque.publish(msg_delta_joint_states)



def main(args=None):
    rclpy.init(args=args)
    foot_contact_detector = FootContactDetector()
    rclpy.spin(foot_contact_detector)
    foot_contact_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()