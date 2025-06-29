import random
import numpy as np

class CoinToss():
    def __init__(self, head_probs, max_episode_steps=30):
        self.head_probs = head_probs
        self.max_episode_steps = max_episode_steps
        self.toss_count = 0


    def __len__(self):
        return len(self.head_probs)
    
    def reset(self):
        self.toss_count = 0

    def step(self, action):
        final = self.max_episode_steps
        if self.toss_count > final:
            raise Exception("The toss count is more than the maximum episode steps.")
        else:
            done = True if self.toss_count == final else False
        
        if action >= len(self.head_probs):
            raise Exception("No.{} coins dpes't exist".format(action))
        else:
            head_prob = self.head_probs[action]
            if random.random() < head_prob:
                reward = 1.0
            else:
                reward = 0.0
        self.toss_count += 1
        return reward, done
        
