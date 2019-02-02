import numpy as np

from rlforge.mixins.base_mixin import BaseMixin
from rlforge.common import Episode


class AgentEvaluatorMX(BaseMixin):
    """Agent Evaluation Mixin.
    Evaluates the agent greedily for `n_evaluate_episodes`
    """

    def __init__(self, n_evaluate_episodes, evaluate_freq, render=False):
        """
        n_evaluate_episodes (int): How many episodes should it be evaluated over
        eval_freq (int): How often (episodes) should the agent be evaluate
        """
        BaseMixin.__init__(self)

        self.evaluate_freq = evaluate_freq
        self.n_evaluate_episodes = n_evaluate_episodes
        self.eval_render = render

        self.post_episode_hooks.append(self.evaluate)

    def evaluate(self, global_episode_ts, episode_data):
        """Evaluate the agent greedily for a few episodes every few episodes.

        post_step_hook Parameters:
            global_step_ts (int)
            episode_data (Episode)
        """
        if global_episode_ts % self.evaluate_freq != 0:
            return

        episode_lengths, episode_returns = [], []

        for i in range(self.n_evaluate_episodes):
            state = self.env.reset()
            episode = Episode(state)

            done = False
            while not done:
                action = self.act(state, greedy=True)
                state_n, reward, done = self.env.step(action)
                episode[-1].append(state_n, reward, action, done)
                if self.eval_render:
                    self.env.render()

                state = state_n

            episode_lengths.append(episode.length)
            episode_returns.append(sum(episode.rewards))

        self.stats.append("eval_mean_episode_lengths", global_episode_ts,
                          np.mean(episode_lengths))

        self.stats.append("eval_mean_episode_returns", global_episode_ts,
                          np.mean(episode_returns))