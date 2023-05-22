import time
import numpy as np
import mujoco as mj
import mujoco_viewer
from sim_mujoco.sim_mujoco_learning.submodules.pid import pid as pid_ctrl
from scipy.spatial.transform import Rotation as R

from threading import Lock
import math 

import csv
import os

def setup_pid(control_rate, kp, ki, kd):
    pid = pid_ctrl()
    pid.pid_set_frequency(control_rate)
    pid.pid_set_gains(kp, ki, kd)
    return pid

class MujocoImitNode:

    ACT_INDS = {
        'L_YAW': 0, 'L_HAA': 1, 'L_HFE': 2, 'L_KFE': 3, 'R_YAW': 4, 'R_HAA': 5, 'R_HFE': 6, 'R_KFE': 7
    }

    def __init__(self, mujoco_xml_path, noisy_imu=False, sim_time_step=0.002, vis_rate=60, visualize=False):
        """Initialized a Mujoco Simulation

        Args:
            mujoco_xml_path (path-like): Location of the XML file to load model from
            sim_time_step (float, optional): Time step for simulation. Defaults to 0.002.
            vis_rate (int, optional): Rate at which to visualize simulation, Hz. Defaults to 60.
            visualize (bool, optional): Whether to visualize the simulation. Defaults to False.
        """
        self.visualize_mujoco = visualize
        self.sim_time_step = sim_time_step
        self.visualization_rate = vis_rate
        self.noisy_imu = noisy_imu
        self.initialization_done = False
        self.goal_pos = [0.0, 0.0]
        self.contact_states = {'R_FOOT': False,
                               'L_FOOT': False}

        self.model = mj.MjModel.from_xml_path(mujoco_xml_path)
        mj.mj_printModel(self.model, 'robot_information.txt')
        self.data = mj.MjData(self.model)
        self.lock = Lock()

        self.time = time.time()
        self.model.opt.timestep = self.sim_time_step
        self.dt = self.model.opt.timestep
        self.R_b_to_I = None
        self.v_b = None
        self.swing_foot_BF_pos = None
        self.stance_foot_BF_pos = None
        self.dcm_desired_BF = None
        self.T_since_contact_right = 0.0
        self.T_since_contact_left = 0.0
        self.T_since_no_contact_right = 0.0
        self.T_since_no_contact_left = 0.0

        self.accel_noise_density = 0.14 * 9.81/1000 # [m/s2 * sqrt(s)]
        self.accel_bias_random_walk = 0.0004 # [m/s2 / sqrt(s)]
        self.gyro_noise_density = 0.0035 / 180 * np.pi # [rad/s * sqrt(s)]
        self.gyro_bias_random_walk = 8.0e-06 # [rad/s / sqrt(s)]
        self.accel_noise_std = self.accel_noise_density / np.sqrt(self.dt)
        self.accel_bias_noise_std = self.accel_bias_random_walk * np.sqrt(self.dt)
        self.gyro_noise_std = self.gyro_noise_density / np.sqrt(self.dt)
        self.gyro_bias_noise_std = self.gyro_bias_random_walk * np.sqrt(self.dt)
        self.accel_bias = np.random.normal(0, self.accel_bias_random_walk * np.sqrt(100), 3)
        self.gyro_bias = np.random.normal(0, self.gyro_bias_random_walk * np.sqrt(100), 3)

        if self.visualize_mujoco == True:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -25
            self.viewer.mjVIS_CONTACTFORCE = True
            self.viewer.render()

        self.name_joints = self.get_joint_names()

        self.q_joints = {}
        for i in self.name_joints:
            self.q_joints[i] = {
                'actual_pos': 0.0,
                'actual_vel': 0.0,
                'desired_pos': 0.0,
                'desired_vel': 0.0,
                'feedforward_torque': 0.0
            }

        self.name_actuators = []
        for i in range(0, self.model.nu):  # skip root
            self.name_actuators.append(mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i))

        self.q_actuator_addr = {}
        for name in self.name_actuators:
            self.q_actuator_addr[name] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, name)

        self.q_pos_addr_joints = {}
        for name in self.name_joints:
            self.q_pos_addr_joints[name] = self.model.jnt_qposadr[mj.mj_name2id(
                self.model, mj.mjtObj.mjOBJ_JOINT, name)]
        
        self.counter = 0
        self.action_shape = (26,)

        self.folder_name = 'sim_mujoco/sim_mujoco_learning/data'
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)

        self.file = open(f'{self.folder_name}/dataset.csv', 'w', newline='')
        self.writer = csv.writer(self.file)
        self.write_controller_dataset_header()
    
    def reset(self, qpos, qvel):
        """Resets the model to a desired initial pose

        Args:
            qpos (array-like): Initial pose of the model
            qvel (array-like): Initial velocity of the model
        """
        self.model.eq_active[0] = 0
        self.data.qpos = qpos
        self.data.qvel = qvel
        self.read_contact_states()
        self.initialization_done = True

    def step(self, action):
        if not self.initialization_done:
            raise RuntimeError("System must initialized before stepping. Call .reset()")

        self.read_contact_states()

        if self.visualize_mujoco is True:
            vis_update_downsampling = int(round(1.0/self.visualization_rate/self.sim_time_step/10))
            if self.counter % vis_update_downsampling == 0:
                self.viewer.render()

        u_ff, q_des, qd_des = MujocoImitNode.parse_action(action)
        self.run_joint_controllers(u_ff, q_des, qd_des)
        mj.mj_step(self.model, self.data)
        self.time += self.dt
        self.counter += 1

        self.R_b_to_I = R.from_quat([self.data.qpos[3], self.data.qpos[4], self.data.qpos[5], self.data.qpos[6]]).as_matrix()
        self.v_b = self.R_b_to_I.T @ self.data.qvel[0:3] # linear vel is in inertial frame

        self.read_contact_states()

        for key, value in self.q_joints.items():
            id_joint_mj = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, key)
            value['actual_pos'] = self.data.qpos[self.model.jnt_qposadr[id_joint_mj]]
            value['actual_vel'] = self.data.qvel[self.model.jnt_dofadr[id_joint_mj]]
            value['actual_acc'] = self.data.qacc[self.model.jnt_dofadr[id_joint_mj]]


        self.ankle_foot_spring('L_ANKLE')
        self.ankle_foot_spring('R_ANKLE')

        self.write_controller_dataset_entry()
        return self.time, self.data.qpos, self.data.qvel


    def run_joint_controllers(self, u_ff, q_des, qd_des):
        """Runs the joint controllers

        Args:
            u_ff (array-like): Feedforward torque to track trajectory
            q_des (array-like): Desired position for motor controller
            qd_des (array-like): Desired velocity for motor controller
        """
        for key, value in self.q_joints.items():
            id_joint_mj = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, key)
            value['actual_pos'] = self.data.qpos[self.model.jnt_qposadr[id_joint_mj]]
            value['actual_vel'] = self.data.qvel[self.model.jnt_dofadr[id_joint_mj]]
            value['actual_acc'] = self.data.qacc[self.model.jnt_dofadr[id_joint_mj]]  
            
            # This is the new way of passing actions
            if key in MujocoImitNode.ACT_INDS.keys():
                value['desired_pos'] = q_des[MujocoImitNode.ACT_INDS[key]]
            if key in MujocoImitNode.ACT_INDS.keys():
                value['desired_vel'] = qd_des[MujocoImitNode.ACT_INDS[key]]
            if key in MujocoImitNode.ACT_INDS.keys() and not math.isnan(u_ff[MujocoImitNode.ACT_INDS[key]]):
                value['feedforward_torque'] = u_ff[MujocoImitNode.ACT_INDS[key]]
            else:
                value['feedforward_torque'] = 0.0

        kp_moteus = 600.0
        Kp = (kp_moteus/(2*math.pi)) * np.ones(self.model.njnt - 1) # exclude root

        for i, (key, value) in enumerate(self.q_joints.items()):
            if key != 'L_ANKLE' and key != 'R_ANKLE':
                feedforward_torque = value['feedforward_torque']
                self.data.ctrl[self.q_actuator_addr[str(key)]] = value['desired_pos']
                self.data.ctrl[self.q_actuator_addr[str(key) + "_VEL"]] = value['desired_vel']
                id_joint_mj = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, key)
                self.data.qfrc_applied[self.model.jnt_dofadr[id_joint_mj]] = feedforward_torque

    staticmethod
    def parse_action(action):
        u_ff = action[:8]
        q_des = action[8:16]
        qd_des = action[16:]
        return u_ff, q_des, qd_des

    def get_IMU_data(self):
        gyro_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, "gyro")
        accel_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, "accelerometer")
        accel = self.data.sensordata[self.model.sensor_adr[accel_id]:self.model.sensor_adr[accel_id] + 3]
        self.accel_bias += np.random.normal(0, self.accel_bias_noise_std, 3)
        
        gyro = self.data.sensordata[self.model.sensor_adr[gyro_id]:self.model.sensor_adr[gyro_id] + 3]
        self.gyro_bias += np.random.normal(0, self.gyro_bias_noise_std, 3)
        if self.noisy_imu:
            accel += self.accel_bias + np.random.normal(0, self.accel_noise_std, 3)
            gyro += self.gyro_bias + np.random.normal(0, self.gyro_noise_std, 3)
        return accel, gyro
    
    def write_controller_dataset_header(self):
        # Dataset: 
            #   time, dt
            #   qpos, qvel, ctrl
            #   joint states
            #   baselink vel BF
            #   imu BF
            #   goal vx_des_BF, vy_des BF
            #   right & left foot: 
            #           t since contact,
            #           t since no contact, 
            #           pos BF
            #   controller: tau_ff, q_j_des, q_j_vel_des
        header_qs = ['q_' + str(i) for i in range(len(self.data.qpos)) ]
        header_qd = ['qd_' + str(i) for i in range(len(self.data.qvel))]
        header_for_tau_ff = ['qfrc_applied_' + str(i) for i in range(len(self.data.qfrc_applied))]
        header_for_ctrl = ['ctrl_' + str(i) for i in range(len(self.data.ctrl))]
        
        header_for_joint_states = [name + '_pos' for name in self.name_joints]
        header_for_joint_states += [name + '_vel' for name in self.name_joints]
        header_for_baselink = ['vel_x_BF', 'vel_y_BF', 'vel_z_BF', 'normal_vec_x_BF', 'normal_vec_y_BF', 'normal_vec_z_BF', 'omega_x', 'omega_y', 'omega_z']
        header_for_imu = ['ax', 'ay', 'az', 'wx', 'wy', 'wz']
        header_for_goal = ['vx_des_BF', 'vy_des_BF']
        header_for_right_foot = ['right_foot_pos_x_BF', 'right_foot_pos_y_BF', 'right_foot_pos_z_BF', 'right_foot_t_since_contact', 'right_foot_t_since_no_contact']
        header_for_left_foot = ['left_foot_pos_x_BF', 'left_foot_pos_y_BF', 'left_foot_pos_z_BF', 'left_foot_t_since_contact', 'left_foot_t_since_no_contact']
        header_for_tau_ff = [name + '_tau_ff' for name in self.name_joints]
        header_for_q_j_des =  [name + '_q_des' for name in self.name_joints]
        header_for_q_j_vel_des = [name + '_q_vel des' for name in self.name_joints]
        
        header = [
            'time', 'dt', *header_qs, *header_qd, *header_for_ctrl,
            *header_for_joint_states, *header_for_baselink, *header_for_imu, *header_for_goal, 
            *header_for_right_foot, 'R_FOOT_contact', *header_for_left_foot, 'L_FOOT_contact',
            *header_for_tau_ff, *header_for_q_j_des, *header_for_q_j_vel_des
        ]
        print(header)
        self.writer.writerow(header)

    def write_controller_dataset_entry(self):
        if self.initialization_done:
            # Dataset: 
            #   time, dt
            #   qpos, qvel, ctrl
            #   joint states
            #   baselink vel BF
            #   imu BF
            #   goal vx_des_BF, vy_des BF
            #   right & left foot: 
            #           t since contact,
            #           t since no contact, 
            #           pos BF
            #   controller: tau_ff, q_j_des, q_j_vel_des
            data_for_qs = self.data.qpos
            data_for_qd = self.data.qvel
            data_for_ctrl = self.data.ctrl
            
            data_for_joint_states = [self.q_joints[name]['actual_pos'] for name in self.name_joints]
            data_for_joint_states += [self.q_joints[name]['actual_vel'] for name in self.name_joints]
            normal_vector_I = np.array([0.0, 0.0, 1.0])
            normal_vector_BF = self.R_b_to_I.T@normal_vector_I
            data_for_baselink = [self.v_b[0], self.v_b[1], self.v_b[2],
                                normal_vector_BF[0], normal_vector_BF[1], normal_vector_BF[2],
                                self.data.qvel[3], self.data.qvel[4], self.data.qvel[5]]
            accel, gyro = self.get_IMU_data()
            data_for_imu = [*accel, *gyro]
            # data_for_goal = [self.dcm_desired_BF[0], self.dcm_desired_BF[1]]
            data_for_goal = [-1, -1]

            if self.contact_states['R_FOOT'] == True:
                # data_for_right_foot = [self.stance_foot_BF_pos[0], self.stance_foot_BF_pos[1], self.stance_foot_BF_pos[2],
                #                     self.T_since_contact_right, self.T_since_no_contact_right]
                data_for_right_foot = [0, 0, 0, self.T_since_contact_right, self.T_since_no_contact_right]
            else:
                # data_for_right_foot = [self.swing_foot_BF_pos[0], self.swing_foot_BF_pos[1], self.swing_foot_BF_pos[2],
                #                     self.T_since_contact_right, self.T_since_no_contact_right]
                data_for_right_foot = [-1, -1, -1, self.T_since_contact_right, self.T_since_no_contact_right]

            if self.contact_states['L_FOOT'] == True:
                # data_for_left_foot = [self.stance_foot_BF_pos[0], self.stance_foot_BF_pos[1], self.stance_foot_BF_pos[2],
                #                     self.T_since_contact_left, self.T_since_no_contact_left]
                data_for_left_foot = [0, 0, 0, self.T_since_contact_left, self.T_since_no_contact_left]
            else:
                # data_for_left_foot = [self.swing_foot_BF_pos[0], self.swing_foot_BF_pos[1], self.swing_foot_BF_pos[2],
                #                     self.T_since_contact_left, self.T_since_no_contact_left]
                data_for_left_foot = [-1, -1, -1, self.T_since_contact_left, self.T_since_no_contact_left]

            data_for_tau_ff = [self.q_joints[name]['feedforward_torque'] for name in self.name_joints]
            data_for_q_j_des = [self.q_joints[name]['desired_pos'] for name in self.name_joints]
            data_for_q_j_vel_des = [self.q_joints[name]['desired_vel'] for name in self.name_joints]

            row_entry = [
                self.time, self.dt, *data_for_qs, *data_for_qd, *data_for_ctrl,
                *data_for_joint_states, *data_for_baselink, *data_for_imu, *data_for_goal, 
                *data_for_right_foot, int(self.contact_states['R_FOOT']), *data_for_left_foot, int(self.contact_states['L_FOOT']),
                *data_for_tau_ff, *data_for_q_j_des, *data_for_q_j_vel_des
            ]
            self.writer.writerow(row_entry)

    def read_contact_states(self):
        self.contact_states['R_FOOT'] = False
        self.contact_states['L_FOOT'] = False

        geom1_list = []
        geom2_list = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            name_geom1 = mj.mj_id2name(
                self.model, mj.mjtObj.mjOBJ_GEOM, contact.geom1)
            name_geom2 = mj.mj_id2name(
                self.model, mj.mjtObj.mjOBJ_GEOM, contact.geom2)
            geom1_list.append(name_geom1)
            geom2_list.append(name_geom2)

        if self.data.ncon != 0:
            if 'L_FOOT' in geom2_list:
                first_entry_idx = geom2_list.index('L_FOOT')
                if geom1_list[first_entry_idx] == 'floor':
                    self.contact_states['L_FOOT'] = True

            if 'R_FOOT' in geom2_list:
                first_entry_idx = geom2_list.index('R_FOOT')
                if geom1_list[first_entry_idx] == 'floor':
                    self.contact_states['R_FOOT'] = True

        if self.contact_states['R_FOOT'] == True:
            self.T_since_contact_right = self.T_since_contact_right + self.dt
            self.T_since_no_contact_right = 0.0

        if self.contact_states['R_FOOT'] == False:
            self.T_since_contact_right = 0.0
            self.T_since_no_contact_right = self.T_since_no_contact_right + self.dt

        if self.contact_states['L_FOOT'] == True:
            self.T_since_contact_left = self.T_since_contact_left + self.dt
            self.T_since_no_contact_left = 0.0

        if self.contact_states['L_FOOT'] == False:
            self.T_since_contact_left = 0.0
            self.T_since_no_contact_left = self.T_since_no_contact_left + self.dt

    def get_joint_names(self):
        self.name_joints = []
        for i in range(1, self.model.njnt):  # skip root
            self.name_joints.append(mj.mj_id2name(
                self.model, mj.mjtObj.mjOBJ_JOINT, i))
        return self.name_joints

    def ankle_foot_spring(self, foot_joint):
        'ankle modelled as a spring damped system'
        K = 0.005
        offset = 0.5
        pitch_error_foot = self.data.qpos[self.q_pos_addr_joints[foot_joint]] - offset
        pitch_torque_setpt = - K * pitch_error_foot

        idx_act = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_ACTUATOR, foot_joint)
        vel_actuator_name = str(foot_joint) + str('_VEL')
        idx_vel_act = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_ACTUATOR, vel_actuator_name)

        self.data.ctrl[idx_act] = pitch_torque_setpt
        self.data.ctrl[idx_vel_act] = 0.0

    def close_writer(self):
        self.file.close()


def main(args=None):
    model_path = "../../install/biped_robot_description/share/biped_robot_description/urdf/custom_robot.mujoco.xml"
    sim_node = MujocoImitNode(model_path, visualize=True)
    qpos0 = np.array([
        -0.0737598007648667, -0.0733845727937982, 0.550748421344813, 0.998831755605091, -0.0471508286450751,
        -0.00841360935195471, 0.00641362070972313, 0.00133506675020533, 0.0259723275317145, 0.811847562710296,
        -1.57679360554637, 0.455073080534268, -0.00480420099908731, 0.0352774013193874, 0.779770272803005,
        -1.10596372383741, 0.350759424411801

    ])
    qvel0 = np.array([
        	-0.23075113830148, -0.0725019859193569, 0.01027672036423, -2.77726862118448, 0.13886494368017,
            -0.446300796379065, 0.18992691041792, -2.79250703895545, 5.33390450874743, -6.63159299878116,
            -6.58420791510949, 0.113297865365863, 3.0988295428688, -1.66376995035463, 1.43100538880684,
            0.23047980128405
    ])
    sim_node.reset(qpos0, qvel0)
    while True:
        sim_node.step(np.zeros(sim_node.action_shape))

if __name__ == '__main__':
    main()
