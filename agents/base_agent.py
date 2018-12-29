import numpy as np

from abc import ABC
from abc import abstractmethod

from rlforge.common.utils import Episode
from rlforge.mixins.base_mixin import BaseMixin



class BaseAgent(ABC):
    """Agent class template
    """
    def __init__(self, env):
        self.env = env

        self.global_step_ts = 0 
        self.global_episode_ts = 0 

        self.pre_episode_hooks = []
        self.post_episode_hooks = []
        self.pre_step_hooks = []
        self.post_step_hooks = [] 


        self.latest_episode = None

    def act(self, state, greedy):
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


    def get_train_batch(self):
        """
        Returns the latest episode as default train_batch
        """
        return self.latest_episode.sarsd_iterator()

    def interact(self, n_episodes=1, render=False):
        """Runs the agent for n_episodes and returns a list of Episodes
        """
        episodes = []
        for i in range(n_episodes):
            #Pre episode hooks
            for pre_episode_hook in self.pre_episode_hooks:
                pre_episode_hook(self.global_episode_ts+1)

            state = self.env.reset()
            episodes.append(Episode(state))

            if render:
                print('Episode {0}.\n'.format(i + 1))
                self.env.render()

            done = False
            while not done:

                #Pre step hook
                for pre_step_hook in self.pre_step_hooks:
                    pre_step_hook(self.global_step_ts+1)

                action = self.act(state, greedy=False)

                state_n, reward, done = self.env.step(action)
                episodes[-1].append(state_n, reward, action, done)
                if render:
                    self.env.render()

                #Post step hook
                self.global_step_ts += 1
                step_data = (state, action, reward, state_n, done)
                for post_step_hook in self.post_step_hooks:
                    post_step_hook(self.global_step_ts, step_data)

                state = state_n

            #Pre episode hooks
            self.global_episode_ts += 1
            for post_episode_hook in self.post_episode_hooks:
                post_episode_hook(self.global_episode_ts, episodes[-1])
        return episodes