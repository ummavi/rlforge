import numpy as np

from tqdm import tqdm
from abc import ABC
from abc import abstractmethod
from collections import defaultdict

from rlforge.common.utils import Episode


class StatsLogger:
    """Simple wrapper over a dict to log agent information
    """

    def __init__(self, experiment=None):
        self._data = defaultdict(lambda: [])
        self.experiment = experiment

    def append(self, key, ts, data):
        self._data[key].append((ts, data))

    def log_scalar(self, key, data):
        ts = len(self._data[key])
        self._data[key].append((ts, data))
        if self.experiment is not None:
            self.experiment.log_scalar(key, data)

    def get(self, key):
        return self._data[key]

    def get_values(self, key):
        # Returns values without the timestamps
        try:
            ts, vals = map(list, zip(*self._data[key]))
        except Exception as e:
            print("Unable to find value with key", key)
            return []
        return vals

    def reset(self):
        self._data = defaultdict(lambda: [])


class BaseAgent(ABC):
    """Agent class template
    """

    def __init__(self, env, experiment=None):
        self.env = env

        self.global_step_ts = 0
        self.global_episode_ts = 0

        self.pre_episode_hooks = []
        self.post_episode_hooks = []
        self.pre_step_hooks = []
        self.post_step_hooks = []
        self.reset_hooks = []

        self.latest_episode = None

        #If a sacred experiment is provided,
        #.. redirect all the log_scalars to that as well.
        self.logger = StatsLogger(experiment)

    @abstractmethod
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

    def interact(self,
                 n_episodes=1,
                 render=False,
                 show_progress=True,
                 progress_metric="episode.returns",
                 progress_metric_lastn=5):
        """Runs the agent for n_episodes and returns a list of Episodes

        Parameters:
            n_episodes: Num. episodes to interact with the environment
            render: Passes render to env() to show active visualization
            show_progress: Uses tqdm to show progress of the interact
            progress_metric: Metric to show latest metic for progress
            progress_metric_lastn: Average the last "n" values of metric
        """
        episodes = []

        pbar = tqdm(range(n_episodes),ascii=True) if show_progress else range(n_episodes)
        for ep in pbar:
            # Pre episode hooks
            for pre_episode_hook in self.pre_episode_hooks:
                pre_episode_hook(self.global_episode_ts + 1)

            state = self.env.reset()
            episodes.append(Episode(state))

            if render:
                print('Episode {0}.\n'.format(i + 1))
                self.env.render()

            done = False
            while not done:

                # Pre step hook
                for pre_step_hook in self.pre_step_hooks:
                    pre_step_hook(self.global_step_ts)

                action = self.act(state, greedy=False)

                state_n, reward, done = self.env.step(action)
                episodes[-1].append(state_n, reward, action, done)
                if render:
                    self.env.render()

                # Post step hook
                self.global_step_ts += 1
                step_data = (state, action, reward, state_n, done)
                for post_step_hook in self.post_step_hooks:
                    post_step_hook(self.global_step_ts, step_data)

                state = state_n

            # Post episode hooks
            self.global_episode_ts += 1
            for post_episode_hook in self.post_episode_hooks:
                post_episode_hook(self.global_episode_ts, episodes[-1])

            self.logger.log_scalar("episode.lengths", episodes[-1].length)
            self.logger.log_scalar("episode.returns",
                                   np.sum(episodes[-1].rewards))

            if show_progress and ep % progress_metric_lastn == 0:
                n = self.logger.get_values(
                    progress_metric)[-progress_metric_lastn:]
                desc = "Latest " + progress_metric + ": " + str(np.mean(n))
                pbar.set_description(desc)

        return episodes

    def reset(self):
        """Resets the weights of the agent and any other parameters
        """
        self.global_step_ts = 0
        self.global_episode_ts = 0
        self.logger.reset()

        for m in self.model_list:
            try:
                m.reset()
            except Exception as e:
                print("reset() not defined for model", m, ". Ignoring")

        for hook in self.reset_hooks:
            hook()
