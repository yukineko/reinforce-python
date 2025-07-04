import numpy as np
import matplotlib.pyplot as plt

class ELAgent():
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.Q = {}
        self.reward_log = []

    def policy(self, s, actions):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(len(actions))
        else:
            print(s)
            if s in self.Q and sum(self.Q[s]) != 0:
                return np.argmax(self.Q[s])
            else:
                return np.random.randint(len(actions))
        return action
    
    def init_log(self):
        self.reward_log = []

    def log(self, reward):
        self.reward_log.append(reward)

    def show_reward_log(self, interval=50, episode=-1):
        if episode > 0:
            rewards = self.reward_log[-interval:]
            mean = np.round(np.mean(rewards), 3)
            std = np.round(np.std(rewards), 3)
            print("At episode {} average reward is {} (+/-{})",
                   episode, mean, std)
        
        else:
            indices = list(range(0, len(self.reward_log), interval))
            means = []
            stds = []
            for i in indices:
                rewards = self.reward_log[i:i + interval]
                means.append(np.mean(rewards))
                stds.append(np.std(rewards))
            means = np.array(means)
            stds = np.array(stds)
            plt.figure()
            plt.title("reward history")
            plt.grid()
            plt.fill_between(indices, 
                             means - stds, 
                             means + stds, 
                             alpha=0.1, 
                             color='g')
            plt.plot(indices, means, 'o-', color='g', 
                     label='Rewards for each {} episodes'.format(interval))
            plt.legend(loc="best")
            plt.show()