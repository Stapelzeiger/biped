from dataclasses import dataclass

import numpy as np
import math

def wrap_circular_value(input_value):
    return (input_value + np.pi) % (2*np.pi) - np.pi


class DoubleIntegrator:
    def __init__(self, dt):
        self.dt = dt
        self.xdim = 2
        self.udim = 1
        self.state = np.zeros(self.xdim)
        self.observation_space = np.zeros(self.udim)
        self.action_space = np.zeros(1)

    def step(self, action, des_pt=None):
        x, x_dot = self.state
        u = action
        assert u.shape == (1,)
        u = u[0]

        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * u

        self.state = np.array([x, x_dot])

        Jx = np.eye(2) + self.dt * np.array([
            [0, 1],
            [0, 0],
        ])

        Ju = self.dt * np.array([
            [0],
            [1],
        ])
        if des_pt is not None:
            costs = (x - des_pt[0]) ** 2 + 0.1 * (x_dot - des_pt[1])**2 + 0.001 * (u**2) # tracking problem
        else:
            costs = x ** 2 + 0.1 * x_dot**2 + 0.001 * (u**2) # reg around 0
        
        done = False

        
        return self._get_obs(), -costs, done, False
    
    def _get_obs(self):
        return self.state
    
    
    def reset(self):
        # reset to random
        self.state = np.random.uniform(-1, 1, self.xdim)
        self.state = np.array([2.0, 0.2])

        return self._get_obs(), {}



class DoubleIntegratorWithPerturbations:
    def __init__(self, dt):
        self.dt = dt
        self.xdim = 2
        self.udim = 1
        self.state = np.zeros(self.xdim)
        self.observation_space = np.zeros(self.udim)
        self.action_space = np.zeros(self.udim)

    def step(self, action, des_pt=None):
        x, x_dot = self.state
        u = action
        assert u.shape == (1,)
        u = u[0]

        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * u - (np.sin(x/10))**2*self.dt

        self.state = np.array([x, x_dot])
        
        x = self.state[0]
        x_dot = self.state[1]

        Jx = np.eye(2) + self.dt * np.array([
            [0, 1],
            [0, 0],
        ])

        Ju = self.dt * np.array([
            [0],
            [1],
        ])
        if des_pt is not None:
            costs = (x - des_pt[0]) ** 2 + 0.1 * (x_dot - des_pt[1])**2 + 0.001 * (u**2) # tracking problem
        else:
            costs = x ** 2 + 0.1 * x_dot**2 + 0.001 * (u**2) # reg around 0
        
        done = False

        return self._get_obs(), -costs, done, False
    
    def _get_obs(self):
        return self.state.reshape(self.xdim)
    
    
    def reset(self):
        # reset to random
        self.state = 2 * np.random.uniform(-1, 1, self.xdim)
        self.state = np.array([2.0, 0.2])
        return self._get_obs(), {}


@dataclass
class AckermannVelDelay:
    def __init__(self, dt):
        self.dt = dt
        self.xdim = 4
        self.udim = 2
        self.state = np.zeros(self.xdim)
        self.observation_space = np.zeros(self.udim)
        self.action_space = np.zeros(self.udim)
        self.tau = 0.1
        self.L = 0.49

    def step(self, action, des_pt=None):
        x, y, theta, vx = self.state
        u_v = action[0]
        steering = action[1]
        c = np.cos(theta)
        s = np.sin(theta)

        theta = theta + self.dt * vx / self.L * np.tan(steering)
        theta = wrap_circular_value(theta)

        x = x + self.dt * vx * c
        y = y + self.dt * vx * s

        vx = vx + self.dt * (1/self.tau * (u_v - vx))

        self.state = np.array([x, y, theta, vx])

        Jx = np.eye(4) + self.dt * np.array([
            [0, 0, -vx*s, c],
            [0, 0,  vx*c, s],
            [0, 0,    0,  np.tan(steering)/self.L],
            [0, 0,    0,  -1/self.tau],
        ])

        Ju = self.dt * np.array([
                            [0,          0],
                            [0,          0],
                            [0.0,        vx/self.L/(np.cos(steering)**2)],
                            [1/self.tau,      0],
                            ])

        costs = False
        done = False
        return self._get_obs(), -costs, done, False

    def _get_obs(self):
        return self.state

    def reset(self):
        # reset to random
        self.state = np.random.uniform(-1, 1, self.xdim)
        return self._get_obs(), {}

system_lookup = dict(
    double_integrator=DoubleIntegrator,
    double_integrator_with_perturbations=DoubleIntegratorWithPerturbations,
    ackermann_vel=AckermannVelDelay,
)
