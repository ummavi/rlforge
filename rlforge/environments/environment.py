import numpy as np
import gym

class GymEnv:
    """
    Creates a wrapper around gym environments
    """
    def __init__(self,gym_env_string):
        self.env = gym.make(gym_env_string)
        try:
            self.n_actions = self.env.action_space.n
            self.action_space = "discrete"

        except Exception:
            #If it doesn't exist, it's probably a continuous env
            self.n_actions = self.env.action_space.shape[0]
            self.action_space = "continuous"

        self.d_observations = list(self.env.observation_space.shape)


    def render(self):
        """Pass the render to the environment"""
        self.env.render()

    def reset(self):
        """Pass reset to the environment"""
        return self.env.reset()

    def step(self, a):
        """Perform one step of the environment and drop the `info`"""
        observation, reward, done, _  = self.env.step(a)
        return (observation, reward, done)

    def seed(self, seed):
        """Seed te environment"""
        self.env.seed(seed)