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

    def step(self, action):
        x, x_dot = self.state
        u = action

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
        costs = x ** 2 + 0.1 * x_dot**2 + 0.001 * (u**2)
        done = False
        if abs(x - 0.0) < 0.01 and abs(x_dot - 0.0) < 0.01:
            done = True
        
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
        self.action_space = np.zeros(1)

    def step(self, action):
        x, x_dot = self.state
        u = action

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
        costs = x ** 2 + 0.1 * x_dot**2 + 0.001 * (u**2)
        
        done = False
        # if abs(x - 0.0) < 0.01 and abs(x_dot - 0.0) < 0.01:
        #     done = True

        return self._get_obs(), -costs, done, False
    
    def _get_obs(self):
        return self.state.reshape(2)
    
    
    def reset(self):
        # reset to random
        self.state = 2 * np.random.uniform(-1, 1, self.xdim)
        self.state = np.array([2.0, 0.2])
        return self._get_obs(), {}


system_lookup = dict(
    double_integrator=DoubleIntegrator,
    double_integrator_with_perturbations=DoubleIntegratorWithPerturbations,
)
