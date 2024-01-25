import numpy as np
import gym
M = 3
L = 1
G = 10
K_energy = 0.1
Kp = 7
Kd = 2

"""Note, modified pendulum environment, won't work off default gym env. Dynamics are wierd. """
def pendulum_expert(obs):
    # cosT = cos(theta), sinT = sin(theta), omega = thetadot
    cosT = obs[0]
    sinT = obs[1]
    omega = obs[2]

    # this same equation as drake, except (1 + cosT)
    # (1 + cosT) is new because adapted to the physics for gym
    desired_energy = 1.1 * M * G * L
    energy = 0.5 * M * pow(L * omega, 2) - (M * G * L * (1 + cosT))
    # energy = 0.5 * (M*L**2) * omega ** 2 - L /2 * (1 - cosT) * M * G
    # energy = (0.5 * M * (L ** 2) * (omega ** 2)) - (M * G * L * cosT)
    # Close to top, use PD control
    if abs(sinT) < 0.2 and cosT > 0.8:
        u = -Kp * sinT - Kd * omega
    else:
        u = K_energy * omega * (desired_energy - energy)
        # try scaling
        u = u
        print(u)
        u = np.clip(u, -2 , 2)

    return [u]


def test_expert():
    env = gym.make("Pendulum-v1", render_mode='human')
    obs = env.reset()
    obs = obs[0] # SJ: IDK what gym version we should use, but under 0.26.2, obs looks like 
    #    (array([cosT , sinT,  omega ], dtype=float32), {}), so it throws index-out-of-range error without obs = obs[0].
    done = False
    while not done:
        u = pendulum_expert(obs)
        res = env.step(u)
        env.render()
        obs = res[0]
        done = res[2]

if __name__ == "__main__":
    test_expert()