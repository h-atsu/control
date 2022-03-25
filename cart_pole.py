import gym
import scipy
import numpy as np
import matplotlib.pyplot as plt

# parameters
num_steps = 3000

# control parameter
Kr = 10  # weight for control
Kq = 0.001  # weight for state


# state matrix
lp = 0.5
mp = 0.1
mk = 1.0
mt = mp + mk
g = 9.8
a = g / (lp * (4.0 / 3 - mp / (mp + mk)))
A = np.array([[0, 1, 0, 0],
              [0, 0, a, 0],
              [0, 0, 0, 1],
              [0, 0, a, 0]])

# input matrix
b = -1 / (lp * (4.0 / 3 - mp / (mp + mk)))
B = np.array([[0], [1 / mt], [0], [b]])

R = Kr * np.eye(1, dtype=int)
Q = Kq * np.eye(4, dtype=int)

# get riccati solver
from scipy import linalg

# solve ricatti equation
P = linalg.solve_continuous_are(A, B, Q, R)

# calculate optimal controller gain
K = np.dot(np.linalg.inv(R),
           np.dot(B.T, P))


def apply_state_controller(K, x):
    # feedback controller
    u = -np.dot(K, x)   # u = -Kx
    if u > 0:
        return 1, u     # if force_dem > 0 -> move cart right
    else:
        return 0, u     # if force_dem <= 0 -> move cart left


# get environment
env = gym.envs.make("CartPole-v0")
env._max_episode_steps = num_steps
env.env.seed(1)     # seed for reproducibility
obs = env.reset()
state_history = []

for i in range(num_steps):
    # get force direction (action) and force value (force)
    action, force = apply_state_controller(K, obs)

    # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
    abs_force = abs(float(np.clip(force, -10, 10)))

    # change magnitute of the applied force in CartPole
    env.env.force_mag = abs_force

    # apply action
    obs, reward, done, _ = env.step(action)
    state_history.append(obs)
    if done:
        print(f'Terminated after {i+1} iterations.')
        break
    env.render()

env.close()

state_history = np.array(state_history).transpose()

labels = [r'$x$', r'$\dot x$', r'$\theta$', r'$\dot \theta$']
for i in range(len(state_history)):
    plt.plot(state_history[i], label=labels[i])
plt.legend()
plt.show()

plt.close()
