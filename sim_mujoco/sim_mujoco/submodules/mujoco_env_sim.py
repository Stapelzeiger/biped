from gymnasium.envs.mujoco import MujocoEnv
import mujoco as mj
import mujoco.viewer

import math
import numpy as np

GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"
IMU_SITE = "imu_location_vectornav"

from brax.training.agents.ppo import checkpoint as ppo_checkpoint
import jax
from jax import numpy as jp
import os
from etils import epath
import sys

jax.config.update('jax_platform_name', 'cpu')

# Load policy.
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
print("Available devices:", jax.devices())
print(str(jax.local_devices()[0]))

RESULTS_FOLDER_PATH ='/home/sorina/Documents/code/biped_hardware/ros2_ws/src/biped/rl_controller/results'

class Biped(MujocoEnv):
    def __init__(self,
                 xml: str,
                 sim_dt: float = 0.002,
                 visualize_mujoco: bool = False,
                 use_RL: bool = False):

        # Initialize the Mujoco environment.
        self.model = mj.MjModel.from_xml_path(xml)
        self.data = mj.MjData(self.model)        
        self.model.opt.timestep = sim_dt
        mj.mj_printModel(self.model, 'robot_information.txt')
        self.dt_sim = self.get_sim_dt() # TODO: mujocoenv has its own DT, but it cannot be modified
        self.use_RL = use_RL

        self.default_q_joints = np.array(self.model.keyframe("home").qpos[7:])
        self.init_q = jp.array(self.model.keyframe("home").qpos)

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

        folders = sorted(os.listdir(RESULTS_FOLDER_PATH))
        latest_folder = folders[-1]
        folders = sorted(os.listdir(epath.Path(RESULTS_FOLDER_PATH) / latest_folder))
        folders = [f for f in folders if os.path.isdir(epath.Path(RESULTS_FOLDER_PATH) / latest_folder / f)]
        if len(folders) == 0:
            raise ValueError(f'No folders found in {epath.Path(RESULTS_FOLDER_PATH) / latest_folder}')
            sys.exit()
        if len(folders) > 1:
            latest_weights_folder = folders[-1]
        else:
            latest_weights_folder = folders
        print(f'Latest weights folder: {latest_weights_folder}')

        path = epath.Path(RESULTS_FOLDER_PATH) / latest_folder / latest_weights_folder
        print(f'Loading policy from: {path}')

        self.policy_fn = ppo_checkpoint.load_policy(path)
        self.jit_policy = jax.jit(self.policy_fn)
        self.rng = jax.random.PRNGKey(1)

        gait_freq = 1.25
        dt = 0.002
        phase_dt = 2 * np.pi * dt * gait_freq
        phase = np.array([0, np.pi])
        self.info = {
            'phase': phase,
            'phase_dt': phase_dt
        }
        self.last_action = np.zeros(8)

        self.counter = 0.0

        # Visualize.
        self.visualize_mujoco = visualize_mujoco
        if self.visualize_mujoco is True:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def init(self,p: list, q: list = [1.0, 0.0, 0.0, 0.0]):
        '''Initializes the robot at a given position, orientation and vel.'''
        # self.model.eq_data[0][0] = p[0]
        # self.model.eq_data[0][1] = p[1]
        # self.model.eq_data[0][2] = 1.0 # one meter above gnd

        # self.data.qpos = [0.0] * self.model.nq
        # self.data.qpos[3] = q[0]
        # self.data.qpos[4] = q[1]
        # self.data.qpos[5] = q[2]
        # self.data.qpos[6] = q[3]

        self.data.qvel = [0.0]* self.model.nv

        # self.data.qpos[0] = p[0]
        # self.data.qpos[1] = p[1]
        # self.data.qpos[2] = self.model.eq_data[0][2]

        # self.data.eq_active[0] = 1
        self.data.qpos = self.init_q.copy()
        mj.mj_step(self.model, self.data)

    def step(self, joint_traj_dict: dict, command: np.array):
        ''' Steps the simulation.'''
        # if self.use_RL == True:
        #     self.run_joint_controllers_RL(joint_traj_dict)
        # else:
        #     self.run_joint_controllers(joint_traj_dict)

        linvel = self.get_sensor_data(LOCAL_LINVEL_SENSOR)
        gyro = self.get_sensor_data(GYRO_SENSOR)
        self._imu_site_id = self.model.site(IMU_SITE).id
        R_gravity_sensor = self.data.site_xmat[self._imu_site_id].reshape(3, 3)
        gravity = R_gravity_sensor.T @ np.array([0, 0, -1]) # TODO: check this
        
        joint_angles = self.data.qpos[7:]
        joint_vel = self.data.qvel[6:]

        phase_tp1 = self.info["phase"] + self.info["phase_dt"]
        self.info["phase"] = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi

        cos = np.cos(self.info["phase"])
        sin = np.sin(self.info["phase"])
        phase = np.concatenate([cos, sin])
        
        # command = np.zeros(3)
        self.state = np.hstack([
            linvel,  # 3
            gyro,  # 3
            gravity,  # 3
            command,  # 3
            joint_angles - self.default_q_joints,  # 12
            joint_vel,  # 12
            self.last_action,  # 12
            phase,
        ])

        obs = {
            'privileged_state': jp.zeros(100),
            'state': jp.array(self.state)
        }
        act_rng, self.rng = jax.random.split(self.rng)
        action, _ = self.jit_policy(obs, act_rng)

        action_to_motor = np.zeros(10)
        action_to_motor[0] = action[0]
        action_to_motor[1] = action[1]
        action_to_motor[2] = action[2]
        action_to_motor[3] = action[3]
        action_to_motor[4] = 0
        action_to_motor[5] = action[4]
        action_to_motor[6] = action[5]
        action_to_motor[7] = action[6]
        action_to_motor[8] = action[7]
        action_to_motor[9] = 0

        motor_targets = self.default_q_joints + action_to_motor * 0.5
        self.data.ctrl = motor_targets
        self.run_joint_controllers(joint_traj_dict)

        # self.ankle_foot_spring('L_ANKLE')
        # self.ankle_foot_spring('R_ANKLE')
        mj.mj_step(self.model, self.data)
        if self.visualize_mujoco is True and self.counter == 20:
            if self.viewer.is_running():
                self.viewer.sync()
                self.counter = 0
        self.counter += 1
        self.last_action = action.copy()
        return self.data.qpos, self.data.qvel

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

    def run_joint_controllers_RL(self, joint_traj_dict: dict):
        if joint_traj_dict is None:
            return
        for key, value in self.q_joints.items():
            if key in joint_traj_dict.keys():
                value['desired_pos'] = joint_traj_dict[key]['pos']

        for key, value in self.q_joints.items():
            if key != 'L_ANKLE' and key != 'R_ANKLE':
                self.data.ctrl[self.q_actuator_addr[str(key)]] = value['desired_pos']

    def run_joint_controllers(self, joint_traj_dict: dict):
        '''Runs the joint controllers.'''
        # if joint_traj_dict is None:
        #     return
        for key, value in self.q_joints.items():
            id_joint_mj = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, key)
            value['actual_pos'] = self.data.qpos[self.model.jnt_qposadr[id_joint_mj]]
            value['actual_vel'] = self.data.qvel[self.model.jnt_dofadr[id_joint_mj]]
            actual_acc = (self.data.qvel[self.model.jnt_dofadr[id_joint_mj]] - self.previous_q_vel[self.model.jnt_dofadr[id_joint_mj]])/self.dt_sim
            value['actual_acc'] = actual_acc
            value['actual_effort'] = self.data.qfrc_actuator[self.model.jnt_dofadr[id_joint_mj]] + self.data.qfrc_applied[self.model.jnt_dofadr[id_joint_mj]]
        #     if key in joint_traj_dict.keys():
        #         value['desired_pos'] = joint_traj_dict[key]['pos']
        #         if joint_traj_dict[key]['vel']:
        #             value['desired_vel'] = joint_traj_dict[key]['vel']
        #         if joint_traj_dict[key]['effort']:
        #             if math.isnan(joint_traj_dict[key]['effort']) is False:
        #                 value['feedforward_torque'] = joint_traj_dict[key]['effort']
        #             else:
        #                 value['feedforward_torque'] = 0.0
        # self.previous_q_vel = self.data.qvel.copy()

        # kp_moteus = 240.0
        # Kp = (kp_moteus/(2*math.pi)) * np.ones(self.model.njnt - 1) # exclude root

        # i = 0
        # for key, value in self.q_joints.items():
        #     if key != 'L_ANKLE' and key != 'R_ANKLE':
        #         # Q: why are we not using a PD controller
        #         error = value['actual_pos'] - value['desired_pos']
        #         actuators_torque = - Kp[i] * error
        #         actuators_vel = value['desired_vel']
        #         feedforward_torque = value['feedforward_torque']
        #         self.data.ctrl[self.q_actuator_addr[str(key)]] = actuators_torque
        #         self.data.ctrl[self.q_actuator_addr[str(key) + "_VEL"]] = actuators_vel
        #         id_joint_mj = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, key)

        #         self.data.qfrc_applied[self.model.jnt_dofadr[id_joint_mj]] = feedforward_torque
        #         self.q_joints[key]['qfrc_actuator'] = self.data.qfrc_actuator[self.model.jnt_dofadr[id_joint_mj]]

        #         self.q_joints[key]['total_tau'] = self.data.qfrc_passive[self.model.jnt_dofadr[id_joint_mj]] + \
        #                                             self.data.qfrc_actuator[self.model.jnt_dofadr[id_joint_mj]] + \
        #                                             self.data.qfrc_applied[self.model.jnt_dofadr[id_joint_mj]]
        #     i = i + 1

    def let_go_of_robot(self):
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
