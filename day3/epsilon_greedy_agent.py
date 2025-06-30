import random
#import numpy as np
class EpsilonGreedyAgent():

    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.V = {}
    
    def policy(self):
        conins = range(len(self.V))
        if random.random() < self.epsilon:
            return random.choice(conins)
        else:
            return np.argmax(self.V)

    def play(self, env):
        N = [0] * len(env)
        self.V = [0] * len(env)

        env.reset()
        done = False
        rewards = []
        while not done:
            selected_coin = self.policy()
            reward, done = env.step(selected_coin)
            print(reward, done)
            rewards.append(reward)
            n = N[selected_coin]
            coin_average = self.V[selected_coin]
            new_average = (coin_average * n + reward) / (n + 1)
            N[selected_coin] += 1
            self.V[selected_coin] = new_average
        return rewards

if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    from coint_toss import CoinToss
    def main():
        env = CoinToss([0.1, 0.5, 0.6, 0.1, 0.9])
        epsilons = [0.0, 0.1, 0.5, 1.0, 0.3]
        game_steps = list(range(10, 310, 10))
        result = {}
        for e in epsilons:
            agent = EpsilonGreedyAgent(epsilon=e)
            means = []
            for s in game_steps:
                env.max_episode_steps = s
                rewards = agent.play(env)
                means.append(np.mean(rewards))
            result["epsilon={}".format(e)] = means
        result["coin toss count"] = game_steps
        result = pd.DataFrame(result)
        result.set_index("coin toss count", inplace=True, drop=True)
        result.plot.line(figsize=(10, 5), title="epsilon-greedy agent")
        plt.show()
    main()