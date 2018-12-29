import numpy as np
import gym

class GymEnv:
    """
    Creates a wrapper around gym environments
    """
    def __init__(self,gym_env_string):
        self.env = gym.make(gym_env_string)

        self.n_actions = self.env.action_space.n
        self.d_observations = list(self.env.observation_space.shape)


    def render(self):
        """Pass the render to the environment"""
        self.env.render()

    def reset(self):
        """Pass reset to the environment"""
        return self.env.reset()

    def step(self,a):
         observation, reward, done, _  = self.env.step(a)
         return (observation, reward, done)
