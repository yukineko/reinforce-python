from planner import Planner
class PolicyIterationPlanner(Planner):
    def __init__(self,env):
        super().__init__(env)
        self.policy = {}

    def initialize(self):
        super().initialize()
        self.policy = {}
        actions = self.env.actions
        states = self.env.states
        for s in states:
            self.policy[s] = {}
            for a in actions:
                self.policy[s][a] = 1.0 / len(actions)
    
    def estimate_by_policy(self, gamma, threshold):
        V = {}
        for s in self.env.states:
            V[s] = 0.0
        
        while True:
            delta = 0.0
            for s in V:
                expected_rewards = []
                for a in self.policy[s]:
                    action_prob = self.policy[s][a]
                    r = 0
                    for prob, next_state, reward in self.transitions_at(s, a):
                        r += action_prob * prob * \
                            (reward + gamma * V[next_state])
                    expected_rewards.append(r)
                value = sum(expected_rewards)
                delta = max(delta, abs(value - V[s]))
                V[s] = value
            if delta < threshold:
                break
        return V

    def plan(self, gamma=0.9, threshold=0.0001):
        self.initialize()
        states = self.env.states
        actions = self.env.actions

        def take_max_action(action_value_dict):
            return max(action_value_dict, key=action_value_dict.get)
        
        while True:
            update_stable = True
            V = self.estimate_by_policy(gamma, threshold)
            self.log.append(self.dict_to_grid(V))
            for s in states:
                policy_action = take_max_action(self.policy[s])
                action_rewards = {}
                for a in actions:
                    r = 0
                    for prob, next_state, reward in self.transitions_at(s, a):
                        r += prob * (reward + gamma * V[next_state])
                    action_rewards[a] = r
                best_action = take_max_action(action_rewards)
                if policy_action != best_action:
                    update_stable = False
                for a in self.policy[s]:
                    prob = 1 if a == best_action else 0
                    self.policy[s][a] = prob
            if update_stable:
                break
        V_grid = self.dict_to_grid(V)
        return V_grid