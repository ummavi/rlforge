"""
Generic agent class that provides 
"""
import numpy as np

class Episode:
    """Simple class to log information an episode.
    """
    def __init__(self,initial_state):
        """initial_state is s_0
        """

        self.length = 0
        self.observations = [initial_state]
        self.rewards = []
        self.actions = []

    def append(self,observation,reward,action):
        """Appends one transition
        """
        self.observations.append(observation)
        self.rewards.append(reward)
        self.actions.append(action)
        self.length+=1

    def sarsa_iterator(self):
        """Return a (s,a,r,s') iterator to the episode
        """
        return zip(self.observations[:-1], self.actions, self.rewards, self.observations[1:])

class Agent:
    """Agent class template
    """
    def __init__(self, env):
        self.env = env

    def act(self, observation, greedy):
        """Template for act() for the agent. Must be overwritten

        Returns:
        Observation_new, action, reward
        """
        raise NotImplementedError()

    def interact(self, n_episodes=1, greedy=False, render=False):
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

                if render:
                    self.env.render()

        return episodes


class RandomAgent(Agent):
    def __init__(self, env):
        Agent.__init__(self, env)

    def act(self, observation, greedy):
        return np.random.choice(self.env.n_actions)