"""
Generic agent class that provides 
"""
import numpy as np


from rlforge.agents.base_agent import BaseAgent

class RandomAgent(BaseAgent):
    def __init__(self, env):
        Agent.__init__(self, env)

    def act(self, observation, greedy):
        return np.random.choice(self.env.n_actions)

    def reset(self):
        returns 