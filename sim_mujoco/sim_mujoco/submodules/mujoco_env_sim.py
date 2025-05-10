import mujoco as mj
import mujoco.viewer

import math
import numpy as np
from collections import deque

import os
from etils import epath
import sys

class Biped():
    def __init__(self,
                 xml: str,
                 sim_dt: float = 0.002,
                 visualize_mujoco: bool = False,
                 use_RL: bool = False,
                 control_delay: float = 0.0):  # Control delay in seconds

        # Initialize the Mujoco environment.
        self.model = mj.MjModel.from_xml_path(xml)
        self.data = mj.MjData(self.model)        
        self.model.opt.timestep = sim_dt
        mj.mj_printModel(self.model, 'robot_information.txt')
        self.dt_sim = self.get_sim_dt() # TODO: mujocoenv has its own DT, but it cannot be modified
        self.use_RL = use_RL

        # Initialize control delay
        self.control_delay = control_delay
        self.delay_buffer_size = max(1, int(control_delay / sim_dt))
        self.control_buffer = deque(maxlen=self.delay_buffer_size)
        self.control_buffer.append({})  # Initialize with empty control dict

        self.q_joints = {}
        self.name_joints = self.get_joint_names()
        for i in self.name_joints:
            self.q_joints[i] = {
                'timestamp': 0.0,
                'actual_pos': 0.0,
                'actual_vel': 0.0,
                'actual_acc': 0.0,
                'actual_effort': 0.0,
                'desired_pos': 0.0,
                'desired_vel': 0.0,
                'feedforward_torque': 0.0,
                'qfrc_actuator': 0.0,
                'total_tau': 0.0,
                'qfrc_passive': 0.0
            }
        self.previous_q_vel = np.zeros(self.get_nv())

        self.time_ = 0.0

        # Visualize.
        self.visualize_mujoco = visualize_mujoco
        if self.visualize_mujoco is True:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def init(self,p: list, q: list = [1.0, 0.0, 0.0, 0.0]):
        '''Initializes the robot at a given position, orientation and vel.'''
        self.model.eq_data[0][0] = p[0]
        self.model.eq_data[0][1] = p[1]
        self.model.eq_data[0][2] = 1.0 # one meter above gnd

        self.data.qpos = [0.0] * self.model.nq
        self.data.qpos[3] = q[0]
        self.data.qpos[4] = q[1]
        self.data.qpos[5] = q[2]
        self.data.qpos[6] = q[3]

        self.zero_the_velocities()

        self.data.qpos[0] = p[0]
        self.data.qpos[1] = p[1]
        self.data.qpos[2] = self.model.eq_data[0][2]

        self.data.eq_active[0] = 1
        mj.mj_step(self.model, self.data)

    def step(self, joint_traj_dict: dict, wrench_perturbation: np.ndarray = None):
        ''' Steps the simulation.'''
        self.set_joint_state()

        # Add current control to buffer
        self.control_buffer.append(joint_traj_dict)

        # Get delayed control from buffer
        delayed_control = self.control_buffer[0]

        if self.use_RL == True:
            self.run_joint_controllers_RL(delayed_control)
        else:
            self.run_joint_controllers(delayed_control)
            self.ankle_foot_spring('L_ANKLE')
            self.ankle_foot_spring('R_ANKLE')

        # Apply an external wrench perturbation to the robot.
        if wrench_perturbation is not None:
            self.apply_wrench_perturbation_to_base_link(wrench_perturbation)

        mj.mj_step(self.model, self.data)
        if self.visualize_mujoco is True:
            if self.viewer.is_running():
                self.viewer.sync()
        self.time_ += self.dt_sim

        return self.data.qpos, self.data.qvel

    def apply_wrench_perturbation_to_base_link(self, wrench_perturbation: np.ndarray):
        '''Applies a wrench perturbation to the robot.'''
        base_link_idx = self.get_base_link_addr()
        self.data.xfrc_applied[base_link_idx] = wrench_perturbation

    def get_q_joints_dict(self):
        '''Returns the joint data.'''
        return self.q_joints

    def get_sensor_data(self, name: str):
        ''' Returns the sensor data.'''
        sensor_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, name)
        sensor_data = self.data.sensordata[self.model.sensor_adr[sensor_id]:self.model.sensor_adr[sensor_id] + 3]
        return sensor_data

    def ankle_foot_spring(self, foot_joint: str):
        '''Control an ankle modelled as a spring damped system '''
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

    def set_joint_state(self):
        for key, value in self.q_joints.items():
            id_joint_mj = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, key)
            value['actual_pos'] = self.data.qpos[self.model.jnt_qposadr[id_joint_mj]]
            value['actual_vel'] = self.data.qvel[self.model.jnt_dofadr[id_joint_mj]]
            actual_acc = (self.data.qvel[self.model.jnt_dofadr[id_joint_mj]] - self.previous_q_vel[self.model.jnt_dofadr[id_joint_mj]])/self.dt_sim
            value['actual_acc'] = actual_acc
            value['actual_effort'] = self.data.qfrc_actuator[self.model.jnt_dofadr[id_joint_mj]] + self.data.qfrc_applied[self.model.jnt_dofadr[id_joint_mj]]

    def run_joint_controllers_RL(self, joint_traj_dict: dict):
        ''' Runs the joint controllers using RL.'''
        if joint_traj_dict is None:
            return
        for key, value in self.q_joints.items():
            if key in joint_traj_dict.keys():
                value['desired_pos'] = joint_traj_dict[key]['pos']

        for key, value in self.q_joints.items():
            # if key != 'L_ANKLE' and key != 'R_ANKLE':
            self.data.ctrl[self.q_actuator_addr[str(key)]] = value['desired_pos']

    def run_joint_controllers(self, joint_traj_dict: dict):
        '''Runs the joint controllers.'''
        if joint_traj_dict is None:
            return
        for key, value in self.q_joints.items():
            id_joint_mj = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, key)
            if key in joint_traj_dict.keys():
                value['desired_pos'] = joint_traj_dict[key]['pos']
                if joint_traj_dict[key]['vel']:
                    value['desired_vel'] = joint_traj_dict[key]['vel']
                if joint_traj_dict[key]['effort']:
                    if math.isnan(joint_traj_dict[key]['effort']) is False:
                        value['feedforward_torque'] = joint_traj_dict[key]['effort']
                    else:
                        value['feedforward_torque'] = 0.0
        self.previous_q_vel = self.data.qvel.copy()

        kp_moteus = 240.0
        Kp = (kp_moteus/(2*math.pi)) * np.ones(self.model.njnt - 1) # exclude root

        i = 0
        for key, value in self.q_joints.items():
            if key != 'L_ANKLE' and key != 'R_ANKLE':
                # Q: why are we not using a PD controller
                error = value['actual_pos'] - value['desired_pos']
                actuators_torque = - Kp[i] * error
                actuators_vel = value['desired_vel']
                feedforward_torque = value['feedforward_torque']
                self.data.ctrl[self.q_actuator_addr[str(key)]] = actuators_torque
                self.data.ctrl[self.q_actuator_addr[str(key) + "_VEL"]] = actuators_vel
                id_joint_mj = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, key)

                self.data.qfrc_applied[self.model.jnt_dofadr[id_joint_mj]] = feedforward_torque
                self.q_joints[key]['qfrc_actuator'] = self.data.qfrc_actuator[self.model.jnt_dofadr[id_joint_mj]]

                self.q_joints[key]['total_tau'] = self.data.qfrc_passive[self.model.jnt_dofadr[id_joint_mj]] + \
                                                    self.data.qfrc_actuator[self.model.jnt_dofadr[id_joint_mj]] + \
                                                    self.data.qfrc_applied[self.model.jnt_dofadr[id_joint_mj]]
            i = i + 1

    def let_go_of_the_robot(self):
        self.model.eq_active0 = 0 # Let go of the robot.
        self.data.eq_active[0] = 0
        
    def zero_the_velocities(self):
        self.data.qvel = [0.0]* self.model.nv

    def read_contact_states(self):
        contact_states = {}
        contact_states['R_FOOT'] = False
        contact_states['L_FOOT'] = False

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
                    contact_states['L_FOOT'] = True

            if 'R_FOOT' in geom2_list:
                first_entry_idx = geom2_list.index('R_FOOT')
                if geom1_list[first_entry_idx] == 'floor':
                    contact_states['R_FOOT'] = True
                    
        return contact_states
        
    def lower_robot(self, step: float):
        self.model.eq_data[0][2] -= 0.5 * step

    def get_nv(self):
        '''number of degrees of freedom = dim(qvel)'''
        return self.model.nv           

    def get_sim_dt(self):
        '''Returns the simulation timestep.'''
        return self.model.opt.timestep

    def get_joint_names(self):
        '''Returns the names of the joints.'''
        self.name_joints = []
        for i in range(1, self.model.njnt):  # skip root
            self.name_joints.append(mj.mj_id2name(
                self.model, mj.mjtObj.mjOBJ_JOINT, i))
        return self.name_joints

    def get_name_actuators(self):
        '''Returns the names of the actuators.'''
        self.name_actuators = []
        for i in range(0, self.model.nu):  # skip root
            self.name_actuators.append(mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i))
        return self.name_actuators
    
    def get_q_actuator_addr(self):
        ''' Returns the address of the actuators.'''
        self.q_actuator_addr = {}
        for name in self.name_actuators:
            self.q_actuator_addr[name] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, name)
        return self.q_actuator_addr
    
    def get_q_pos_addr_joints(self):
        ''' Returns the address of the position of the joints.'''
        self.q_pos_addr_joints = {}
        for name in self.name_joints:
            self.q_pos_addr_joints[name] = self.model.jnt_qposadr[mj.mj_name2id(
                self.model, mj.mjtObj.mjOBJ_JOINT, name)]

    def get_base_link_addr(self):
        ''' Returns the address of the base link.'''
        return self.data.body("base_link").id