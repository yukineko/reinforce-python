from el_agent import ELAgent
import gym
import numpy as np
class Actor(ELAgent):
    def __init__(self, epsilon=-1):
        super().__init__(epsilon)
        nrow = gym.observation_space.n
        ncol = gym.action_space.n
        self.actions = list(range(gym.action_space.n))
        self.Q = np.random.uniform(0, 1, nrow * ncol).reshape((nrow, ncol))

    def sftmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    
    def policy(self, s)
        a = np.random.choice(self,actions, 1. p=self.softmax(self.Q[s]))
        return a[0]
    