from gymnasium.envs.mujoco import MujocoEnv
import mujoco as mj
import mujoco.viewer
from trajectory_msgs.msg import JointTrajectory

import math
import numpy as np

class Biped(MujocoEnv):
    def __init__(self,
                 xml: str,
                 sim_dt: float = 0.002,
                 visualize_mujoco: bool = False):

        # Initialize the Mujoco environment.
        self.model = mj.MjModel.from_xml_path(xml)
        self.data = mj.MjData(self.model)        
        self.model.opt.timestep = sim_dt
        mj.mj_printModel(self.model, 'robot_information.txt')
        self.dt_sim = self.get_sim_dt() # TODO: mujocoenv has its own DT, but it cannot be modified

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

        self.data.qvel = [0.0]* self.model.nv

        self.data.qpos[0] = p[0]
        self.data.qpos[1] = p[1]
        self.data.qpos[2] = self.model.eq_data[0][2]

        self.data.eq_active[0] = 1
        mj.mj_step(self.model, self.data)
    
    def step(self, is_valid_traj_msg: bool, joint_traj_msg: JointTrajectory):
        '''Steps the simulation.'''
        self.run_joint_controllers(is_valid_traj_msg, joint_traj_msg)
        self.ankle_foot_spring('L_ANKLE')
        self.ankle_foot_spring('R_ANKLE')
        mj.mj_step(self.model, self.data)
        if self.visualize_mujoco is True:
            if self.viewer.is_running():
                self.viewer.sync()
        return self.data.qpos, self.data.qvel

    def get_q_joints_dict(self):
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

    def run_joint_controllers(self, is_valid_traj_msg: bool, joint_traj_msg: JointTrajectory):
        '''Runs the joint controllers.'''
        # TODO: remove the JointTrajectory dependency
        if is_valid_traj_msg is False:
            return
        for key, value in self.q_joints.items():
            id_joint_mj = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, key)
            value['actual_pos'] = self.data.qpos[self.model.jnt_qposadr[id_joint_mj]]
            value['actual_vel'] = self.data.qvel[self.model.jnt_dofadr[id_joint_mj]]
            actual_acc = (self.data.qvel[self.model.jnt_dofadr[id_joint_mj]] - self.previous_q_vel[self.model.jnt_dofadr[id_joint_mj]])/self.dt_sim
            value['actual_acc'] = actual_acc
            value['actual_effort'] = self.data.qfrc_actuator[self.model.jnt_dofadr[id_joint_mj]] + self.data.qfrc_applied[self.model.jnt_dofadr[id_joint_mj]]
            if key in joint_traj_msg.joint_names:
                id_joint_msg = joint_traj_msg.joint_names.index(key)
                value['desired_pos'] = joint_traj_msg.points[0].positions[id_joint_msg]
                if joint_traj_msg.points[0].velocities:
                    value['desired_vel'] = joint_traj_msg.points[0].velocities[id_joint_msg]
                if joint_traj_msg.points[0].effort:
                    if math.isnan(joint_traj_msg.points[0].effort[id_joint_msg]) is False:
                        value['feedforward_torque'] = joint_traj_msg.points[0].effort[id_joint_msg]
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

    def let_go_of_robot(self):
        self.model.eq_active0 = 0 # let go of the robot
        self.data.eq_active[0] = 0 # let go of the robot
        
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
