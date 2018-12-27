import numpy as np

from abc import ABC
from abc import abstractmethod

from rlforge.common.utils import Episode

class BaseAgent(ABC):
    """Agent class template
    """
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def act(self, observation, greedy):
        """Template for act() for the agent. Must be overwritten

        Returns:
        action the agent is to perform.
        """
        raise NotImplementedError()

    @abstractmethod
    def learn(self, data, **kwargs):
        """Performs an update on one "batch" of data
        data can be a list of episodes or a batch of transitions.
        """
        raise NotImplementedError()


    def save(self, dirname):
        """Save agent state to path
        """
        pass

    def load(self, dirname):
        """Load agent state from path
        """
        pass 

        
    def interact_episodes(self, n_episodes=1, greedy=False, render=False):
        """Runs the agent for n_episodes and returns a list of Episodes
        """
        episodes = []
        for i in range(n_episodes):
            obs = self.env.reset()
            episodes.append(Episode(obs))

            if render:
                print('Episode {0}.\n'.format(i + 1))
                self.env.render()

            done = False
            while not done:
                action = self.act(obs, greedy)
                obs, reward, done = self.env.step(action)
                episodes[-1].append(obs, reward, action)
                episodes[-1].finish_episode()
                if render:
                    self.env.render()
        return episodes
