import numpy as np
import gym
DES_ENERGY = 0.1
M = 1
L = 1
G = 10
K_energy = 0.5
Kp = 7
Kd = 2

"""Note, modified pendulum environment, won't work off default gym env. Dynamics are wierd. """
def pendulum_expert(obs):
    cosT = obs[0]
    sinT = obs[1]
    omega = obs[2]

    energy = 0.5 * (M*L**2) * omega ** 2 - L /2 * (1 - cosT) * M * G
    # Close to top, use PD control
    if abs(sinT) < 0.2 and cosT > 0.8:
        u = -Kp * sinT - Kd * omega
    else:
        u = np.clip(K_energy*(DES_ENERGY - energy) * np.sign(omega), -2, 2)

    return [u]


def test_expert():
    env = gym.make("Pendulum-v1")
    obs = env.reset()
    done = False
    while not done:
        u = pendulum_expert(obs)
        res = env.step(u)
        env.render()
        obs = res[0]
        done = res[2]

if __name__ == "__main__":
    test_expert()