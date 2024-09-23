#!/usr/bin/env python3

import rclpy
import time
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import numpy as np
import threading
from rclpy.parameter import Parameter
from rcl_interfaces.srv import SetParameters

TIME_PERIOD = 0.01
VEL_MAX = 0.3
COUNTER_TRIGGER = 20
EPSILON = 0.0001

LOW_TORQUE = 3.0
HIGH_TORQUE = 5.0


class JointCalibration(Node):
    def __init__(self):
        super().__init__('joint_calibration_publisher')
        # this will get moteus_drv share folder.
        # on the rasberrypi: /home/biped-raspi/biped_ws/install/moteus_drv/share/moteus_drv
        self.ws_share_folder = self.declare_parameter('install_folder', rclpy.Parameter.Type.STRING).value # todo use ros tooling for this
        self.get_logger().info(f'Workspace share folder: {self.ws_share_folder}')
        list_motors = self.declare_parameter('joints', rclpy.Parameter.Type.STRING_ARRAY).value
        self.get_logger().info(f'List of motors: {list_motors}')

        self.joints_dict = { # TODO populate this dict from the config params.yaml
            'joint_names': list_motors,
            'is_calibrated': [False]*len(list_motors),
            'joint_pos': [None]*len(list_motors),
            'joint_vel': [None]*len(list_motors),
            'joint_effort': [None]*len(list_motors),
            'center_pos': [None]*len(list_motors),
            'limit': [None]*len(list_motors),
            'lower_limit': [None]*len(list_motors),
            'initial_pos': [None]*len(list_motors),
            'centering_pos_deg': [None]*len(list_motors),
            'trigger_effort': [None]*len(list_motors),
            'direction': [None]*len(list_motors),
            'vel_max': [None]*len(list_motors),
            'set_high_torque_param': [False]*len(list_motors),
            'set_low_torque_param': [False]*len(list_motors),
        }

        for i, joint_name in enumerate(self.joints_dict['joint_names']):
            # offset param.
            offset_param_str = f'{joint_name}/offset'
            self.declare_parameter(offset_param_str, 0.0)

            # max_torque param.
            max_torque_param_str = f'{joint_name}/max_torque'
            self.declare_parameter(max_torque_param_str, LOW_TORQUE)

            # since not everything is centered at 0, we need to know the centering factor.
            self.joints_dict['centering_pos_deg'][i] = self.declare_parameter(f'{joint_name}/calib/centering_pos_deg', rclpy.Parameter.Type.DOUBLE).value
            self.joints_dict['trigger_effort'][i] = self.declare_parameter(f'{joint_name}/calib/trigger_effort', rclpy.Parameter.Type.DOUBLE).value
            self.joints_dict['direction'][i] = self.declare_parameter(f'{joint_name}/calib/direction', rclpy.Parameter.Type.DOUBLE).value

            self.joints_dict['vel_max'][i] = VEL_MAX * self.joints_dict['direction'][i]

        self.lock = threading.Lock()

        self.counter = 0
        self.counter_ramp_center = 0
        self.write_offsets = False

        self.setpt_pos = None
        self.setpt_vel = None

        self.moteus_set_param = self.create_client(SetParameters, '/moteus/set_parameters')
        while not self.moteus_set_param.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.moteus_param_requests = []

        self.update_moteus_parameter('global_max_torque', HIGH_TORQUE)

        for joint_name in self.joints_dict['joint_names']:
            self.update_moteus_parameter(f'{joint_name}/offset', 0.0)
        while len(self.moteus_param_requests) > 0:
            self.check_moteus_param_requests()
            self.get_logger().info('Waiting for all offsets to be set to 0')
            rclpy.spin_once(self, timeout_sec=1)
            time.sleep(0.1)
        # wait for some time to make sure motors are relative to new parameters
        start = time.time()
        while time.time() - start < 5:
            self.get_logger().info('Waiting for motors to be relative to new parameters')
            rclpy.spin_once(self, timeout_sec=1)
            time.sleep(0.1)

        self.pub_trajectory = self.create_publisher(JointTrajectory, 'joint_trajectory', 10)
        self.joint_states_sub = self.create_subscription(JointState, 'joint_states', self.joint_states_callback, 10)
        self.timer_period = TIME_PERIOD # seconds

        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def update_moteus_parameter(self, name_param, value):
        req = SetParameters.Request()
        req.parameters = [Parameter(name=name_param, value=value).to_parameter_msg()]
        self.moteus_param_requests.append(self.moteus_set_param.call_async(req))

    def check_moteus_param_requests(self):
        requests_open = []
        for i, req in enumerate(self.moteus_param_requests):
            if req.done():
                try:
                    response = req.result()
                except Exception as e:
                    self.get_logger().error(f'Service call failed {str(e)}')
                else:
                    self.get_logger().info(f'Parameter updated')
            else:
                requests_open.append(req)
        self.moteus_param_requests = requests_open

    def joint_states_callback(self, msg):
        with self.lock:
            for _, joint in enumerate(self.joints_dict['joint_names']):
                if joint in msg.name:
                    idx = msg.name.index(joint)
                    self.joints_dict['joint_pos'][idx] = msg.position[idx]
                    self.joints_dict['joint_vel'][idx] = msg.velocity[idx]
                    self.joints_dict['joint_effort'][idx] = msg.effort[idx]
                    self.joints_dict['initial_pos'][idx] = msg.position[idx]


    def get_calibration_setpt_msg(self, joint, idx):
        with self.lock:

            joint_effort = self.joints_dict['joint_effort'][idx]
            # if the joint is not moving and the center pos is not determined, record the position
            if np.abs(joint_effort) > self.joints_dict['trigger_effort'][idx] and self.joints_dict['center_pos'][idx] is None:
                if  self.counter > COUNTER_TRIGGER: # going forward
                    self.get_logger().info(f'Joint {joint} is not moving, record position')
                    self.joints_dict['limit'][idx] = self.joints_dict['joint_pos'][idx]
                    self.get_logger().info(f'Limit: {self.joints_dict["limit"][idx]}')
                    self.counter = 0
                    self.get_logger().info(f'Effort: {joint_effort}')
                    self.joints_dict['center_pos'][idx] = self.joints_dict['limit'][idx] - self.joints_dict['direction'][idx] * np.radians(self.joints_dict['centering_pos_deg'][idx])
                    self.get_logger().info(f'Center pos: {self.joints_dict["center_pos"][idx]}')

            # center pos is determined, move the joint to the center pos
            if self.joints_dict['center_pos'][idx] is not None:
                if np.abs(self.joints_dict['joint_pos'][idx] - self.joints_dict['center_pos'][idx]) > EPSILON:
                    # drive the robot to the center position by ramping down position
                    self.setpt_pos = self.joints_dict["limit"][idx] - self.joints_dict['vel_max'][idx]*self.counter_ramp_center*TIME_PERIOD
                    self.setpt_vel = -self.joints_dict['vel_max'][idx]
                    self.counter_ramp_center += 1

                # verify the center position was achieved
                if np.abs(self.joints_dict['joint_pos'][idx] - self.joints_dict['center_pos'][idx]) < EPSILON:
                    self.joints_dict['is_calibrated'][idx] = True
                    self.counter = 0
                    self.counter_ramp_center = 0
                    self.get_logger().info(f'Joint {joint} is calibrated')
                    self.get_logger().info(f'Center Position achieved: {self.joints_dict["joint_pos"][idx]}')

                # check for overshooting (TODO: make this better)
                if self.joints_dict['vel_max'][idx] > 0 and self.joints_dict['joint_pos'][idx] < self.joints_dict['center_pos'][idx]:
                    self.get_logger().info(f'Overshooting, stop the joint')
                    self.setpt_vel = 0
                    self.joints_dict['is_calibrated'][idx] = True
                    # bring back to center
                    self.joints_dict['joint_pos'][idx] = self.joints_dict['center_pos'][idx]
                    self.get_logger().info(f'Joint position achieved: {self.joints_dict["joint_pos"][idx]}')
                    self.counter = 0
                    self.counter_ramp_center = 0

                if self.joints_dict['vel_max'][idx] < 0 and self.joints_dict['joint_pos'][idx] > self.joints_dict['center_pos'][idx]:
                    self.get_logger().info(f'Overshooting, stop the joint')
                    self.setpt_vel = 0
                    self.joints_dict['is_calibrated'][idx] = True
                    # bring back to center
                    self.joints_dict['joint_pos'][idx] = self.joints_dict['center_pos'][idx]
                    self.get_logger().info(f'Joint position achieved: {self.joints_dict["joint_pos"][idx]}')
                    self.counter = 0
                    self.counter_ramp_center = 0

            # center pos was not determined, move the joint
            if self.joints_dict['center_pos'][idx] is None:
                self.setpt_pos = self.joints_dict['vel_max'][idx]*self.counter*(TIME_PERIOD*0.1) + self.joints_dict['initial_pos'][idx]
                self.setpt_vel = self.joints_dict['vel_max'][idx]

            if self.setpt_pos is None or self.setpt_vel is None:
                self.get_logger().info('Setpoint position or velocity is None')
                return

            joint_traj_pt_msg = JointTrajectoryPoint()
            joint_traj_pt_msg.positions.append(self.setpt_pos)
            joint_traj_pt_msg.velocities.append(self.setpt_vel)
            joint_traj_pt_msg.effort.append(0)

            return joint_traj_pt_msg

    def timer_callback(self):
        self.check_moteus_param_requests()

        if None in self.joints_dict['joint_pos'] or \
            None in self.joints_dict['joint_vel'] or \
            None in self.joints_dict['joint_effort'] or \
            None in self.joints_dict['initial_pos']:
            self.get_logger().info('No joint states received yet')
            return

        calibration_done = self.joints_dict['is_calibrated'] == [True]*len(self.joints_dict['joint_names'])
        if calibration_done:
            self.get_logger().info('All motors are calibrated')

            # set the joint traj to nan
            msg = JointTrajectory()
            msg.header.stamp = self.get_clock().now().to_msg()
            for i, joint in enumerate(self.joints_dict['joint_names']):
                msg.joint_names.append(joint)
                joint_traj_pt_msg = JointTrajectoryPoint()
                joint_traj_pt_msg.positions.append(np.nan)
                joint_traj_pt_msg.velocities.append(np.nan)
                joint_traj_pt_msg.effort.append(np.nan)
                msg.points.append(joint_traj_pt_msg)
            self.pub_trajectory.publish(msg)

            if self.write_offsets == False:
                self.write_offsets = True
                self.get_logger().info('Writing offsets to file')
                # this will save to share folder which is symlinked to the original file.
                file_name = self.ws_share_folder + '/config/calibration.yaml'

                with open(file_name, 'w') as output_file:
                    output_file.writelines('/**:\n')
                    output_file.writelines('  ros__parameters:\n')

                    for i, joint in enumerate(self.joints_dict['joint_names']):
                        self.get_logger().info('i: ' + str(i))
                        offset_pos = self.joints_dict['joint_pos'][i]
                        offset = f'    {joint}/offset: {offset_pos}\n'
                        output_file.writelines(offset)
                        self.get_logger().info(f'Joint {joint} position after calibration: {self.joints_dict["joint_pos"][i]}')
            return # calibration done

        # Take the first uncalibrated joint and calibrate it
        for i, joint in enumerate(self.joints_dict['joint_names']):
            if self.joints_dict['is_calibrated'][i] == False:
                # self.get_logger().info(f'Calibrating joint {joint}')

                # Set max torque to a low value for that particular joint.
                if self.joints_dict['set_low_torque_param'][i] == False:
                    max_torque_per_joint_param_name = f'{joint}/max_torque'
                    self.update_moteus_parameter(max_torque_per_joint_param_name, LOW_TORQUE)
                    self.joints_dict['set_low_torque_param'][i] = True

                # Calibrate.
                msg = JointTrajectory()
                msg.header.stamp = self.get_clock().now().to_msg()
                joint_traj_pt_msg = self.get_calibration_setpt_msg(joint, i)
                msg.joint_names.append(joint)
                msg.points.append(joint_traj_pt_msg)
                break

        for i, joint in enumerate(self.joints_dict['joint_names']):
            if self.joints_dict['is_calibrated'][i] == True:

                # Set max torque to a high value for the joints that are already calibrated.
                if self.joints_dict['set_high_torque_param'][i] == False:
                    max_torque_per_joint_param_name = f'{joint}/max_torque'
                    self.update_moteus_parameter(max_torque_per_joint_param_name, HIGH_TORQUE)
                    self.joints_dict['set_high_torque_param'][i] = True

                msg.joint_names.append(joint)
                joint_traj_pt_msg = JointTrajectoryPoint()
                joint_traj_pt_msg.positions.append(self.joints_dict['center_pos'][i])
                joint_traj_pt_msg.velocities.append(0)
                joint_traj_pt_msg.effort.append(0)
                msg.points.append(joint_traj_pt_msg)
        self.pub_trajectory.publish(msg)
        self.counter += 1

def main(args=None):
    rclpy.init(args=args)
    joint_trajectory_publisher = JointCalibration()
    rclpy.spin(joint_trajectory_publisher)
    joint_trajectory_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()