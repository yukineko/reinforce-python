import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gym
from gym.envs.registration import register
register(id='FrozenLakeEasy-v0',entry_point='gym.envs.toy_text:FrozenLakeEnv',
         kwargs={'is_slippery': False})

def show_q_value(Q):
    env = gym.make('FrozenLakeEasy-v0')
    nrow = env.unwrapped.nrow
    ncol = env.unwrapped.ncol
    state_size = 3
    q_nrow = nrow * state_size
    q_ncol = ncol * state_size
    reward_map = np.zeros((q_nrow, q_ncol))
    for r in range(nrow):
        for c in range(ncol):
            s = r * nrow + c
            state_exists = False
            if isinstance(Q, dict) and s in Q:
                state_exists = True
            elif isinstance(Q, (np.ndarray, np.generic)) and s < Q.shape[0]:
                state_exists = True
            if state_exists:
                _r = 1 + (nrow - 1 - r) * state_size
                _c = 1 + c * state_size
                reward_map[_r][_c - 1] = Q[s][0]
                reward_map[_r - 1][_c] = Q[s][1]
                reward_map[_r][_c + 1] = Q[s][2]
                reward_map[_r + 1][_c] = Q[s][3]
                reward_map[_r][_c] = np.mean(Q[s])
        
