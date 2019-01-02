import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

from rlforge.agents.base_agent import BaseAgent
from rlforge.mixins.policies import SoftmaxPolicyMX


class REINFORCEAgent(SoftmaxPolicyMX, BaseAgent):
    """
    Simple Episodic REINFORCE (Williams92) agent with discrete actions
    and a softmax policy.

    """

    def __init__(self, env, model, policy_learning_rate,
                 baseline=None, baseline_learning_rate=None,
                 gamma=0.9, entropy_coeff=0.0):
        """
        Parameters:
        model: A callable model with the final layer being identity.
        baseline (None/Value Function): Optional baseline function
        """
        BaseAgent.__init__(self, env)
        SoftmaxPolicyMX.__init__(self)

        self.model = model
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.opt = tf.train.AdamOptimizer(policy_learning_rate)

        self.post_episode_hooks.append(self.learn)

        if baseline is None:
            # If baseline isn't set, use a constant 0 baseline.
            self.baseline = lambda x: np.zeros(len(x))
        else:
            self.baseline = baseline
            self.baseline_opt = tf.train.AdamOptimizer(baseline_learning_rate)
            self.post_episode_hooks.append(self.learn_baseline)

    def get_discounted_returns(self, rewards):
        """ Return discounted rewards
        """

        discount_multipliers = np.power(
            self.gamma, np.arange(len(rewards)))
        discounted_returns = []
        for i in range(len(rewards)):
            future_rewards = np.float32(rewards[i:])
            discount_multipliers = discount_multipliers[:len(future_rewards)]
            discounted_return = np.sum(future_rewards * discount_multipliers)
            discounted_returns.append(discounted_return)

        return np.float32(discounted_returns)

    def learn_baseline(self, global_episode_ts, episode_data):
        """ Perform a step of training for the Monte-Carlo Value-function
        estimator.

        post_step_hook Parameters:
            global_step_ts (int)
            episode_data (Episode)
        """
        discounted_returns = self.get_discounted_returns(episode_data.rewards)
        states = np.float32(episode_data.observations[:-1])

        with tf.GradientTape() as tape:
            preds = self.baseline(states)
            discounted_returns = np.reshape(discounted_returns, (-1, 1))
            losses = tf.losses.mean_squared_error(discounted_returns, preds)

            grads = tape.gradient(losses, self.baseline.trainable_weights)
            self.baseline_opt.apply_gradients(zip(grads,
                                        self.baseline.trainable_weights))

    def learn(self, global_episode_ts, episode_data):
        """Perform one step of learning with a batch of data

        post_step_hook Parameters:
            global_step_ts (int)
            episode_data (Episode)
        """

        discounted_returns = self.get_discounted_returns(episode_data.rewards)
        states, actions = episode_data.observations[:-1], episode_data.actions

        discounted_returns = discounted_returns - self.baseline(states)
        with tf.GradientTape() as tape:
            probs = tf.nn.softmax(self.model(states))
            logprobs = tf.log(probs)

            selected_logprobs = logprobs * \
                tf.one_hot(actions, self.env.n_actions)
            selected_logprobs = tf.reduce_sum(selected_logprobs, axis=-1)

            average_entropy = tf.reduce_mean(
                tf.reduce_sum(probs * logprobs, axis=1))

            losses = -tf.reduce_sum(selected_logprobs * discounted_returns)
            losses += self.entropy_coeff * average_entropy

            grads = tape.gradient(losses, self.model.trainable_weights)
            self.opt.apply_gradients(zip(grads,
                                         self.model.trainable_weights))

        self.stats.append("episode_average_entropy",
                          global_episode_ts, average_entropy)
        self.stats.append("episode_losses", global_episode_ts, losses)


if __name__ == '__main__':
    from tqdm import tqdm
    from rlforge.common.policy_functions import PolicyNetworkDense
    from rlforge.common.value_functions import VNetworkDense
    from rlforge.environments.environment import GymEnv

    def train(agent, n_episodes):
        # Simple train function using tqdm to show progress
        pbar = tqdm(range(n_episodes))
        for i in pbar:
            _ = agent.interact(1)
            if i % 5 == 0:
                last_5_rets = agent.stats.get_values("episode_returns")[-5:]
                pbar.set_description("Latest return: " +
                                     str(np.mean(last_5_rets)))

    env = GymEnv('CartPole-v0')

    np.random.seed(0)
    tf.set_random_seed(0)
    env.env.seed(0)

    policy = PolicyNetworkDense(env.n_actions, dict(layer_sizes=[64, 64],
                                                    activation="tanh"))
    value_baseline = VNetworkDense(dict(layer_sizes=[64, 64],
                                        activation="tanh"))
    agent = REINFORCEAgent(env, policy,
                           policy_learning_rate=0.001,
                           baseline=value_baseline,
                           baseline_learning_rate=0.001,
                           gamma=0.9, entropy_coeff=3)
    train(agent, 500)
    print("Average Return (Train)", np.mean(
        agent.stats.get_values("episode_returns")))
