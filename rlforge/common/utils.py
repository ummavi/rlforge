import numpy as np
from scipy.signal import savgol_filter


def uniform_random_choice(choices):
    """Make a random choice from choices.
    """
    random_index = np.random.randint(len(choices))
    return choices[random_index]


def smooth_signal(data, window_size=5, poly_order=2):
    """Smooth 1d signal with a moving average
    """
    return savgol_filter(data, window_size, poly_order)


def discounted_returns(rewards, gamma, n_steps=None):
    """ Return discounted rewards

    Parameters:
    rewards (np.array/list): Sequence of rewards
    gamma (float): Discount factor
    n_steps (int): n_step discounted returns. Default is returns till term.
    """
    discount_multipliers = np.power(gamma, np.arange(len(rewards)))
    discounted_returns = []
    for i in range(len(rewards)):
        # If n_steps is None, index till the end of reward list
        tend_index = None if n_steps is None else i+n_steps
        future_rewards = np.float32(rewards[i:tend_index])
        discount_multipliers = discount_multipliers[:len(future_rewards)]
        discounted_return = np.sum(future_rewards * discount_multipliers)
        discounted_returns.append(discounted_return)
    return discounted_returns


class Episode:
    """Simple class to log information an episode.
    """

    def __init__(self, initial_state):
        """initial_state is s_0
        """

        self.length = 0
        self.observations = [initial_state]
        self.rewards = []
        self.actions = []
        self.dones = []
        self.terminated = False

    def append(self, observation, reward, action, done):
        """Appends one transition
        """
        self.observations.append(observation)
        self.rewards.append(reward)
        self.actions.append(action)
        self.dones.append(done)
        self.length += 1

        if done:
            self.finish_episode()

    def sarsd_iterator(self):
        """Return a (s,a,r,s',done) iterator to the episode
        """
        return zip(self.observations[:-1], self.actions,
                   self.rewards, self.observations[1:],
                   self.dones)

    def finish_episode(self):
        """Marks an episode as having terminated.
        """
        self.terminated = True
