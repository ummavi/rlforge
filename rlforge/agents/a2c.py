from rlforge.modules.policies import SoftmaxPolicyMX, GaussianPolicyMX
from rlforge.agents.base_agent import BaseAgent
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()


class A2CAgent(SoftmaxPolicyMX, BaseAgent):
    """
    A2C paper implementation as described in
    (https://blog.openai.com/baselines-acktr-a2c/
    https://arxiv.org/abs/1602.01783)

    Note: Instead of doing parallel rollouts, gradients are batched
    and updated every `n_workers` episodes.

    See examples/a2c.py for example agent

    TODO:
    * Parallel runners
    * GAE (https://arxiv.org/abs/1506.02438)
    * Shared network layers
    """

    def __init__(self,
                 env,
                 model,
                 policy_learning_rate,
                 v_function,
                 gamma=0.9,
                 entropy_coeff=3,
                 n_workers=2):
        """
        Parameters:
        model: A callable model with the final layer being identity.
        v_function: A callable model of the state-value function
        """
        BaseAgent.__init__(self, env)
        SoftmaxPolicyMX.__init__(self)

        self.gamma = gamma
        self.n_workers = n_workers
        self.entropy_coeff = entropy_coeff

        self.model = model
        self.opt = tf.train.AdamOptimizer(policy_learning_rate)
        self.post_episode_hooks.append(self.learn)

        self.v_function = v_function
        self.post_episode_hooks.append(self.learn_value)

        self.accumulated_grads = None

        self.model_list = [self.model, self.v_function]

    def get_discounted_returns(self, rewards):
        """ Return discounted rewards

        Note: Returns are *inclusive* of current timestep
              G_t = r_t + r_{t+1}....
        """
        discount_multipliers = np.power(self.gamma, np.arange(len(rewards)))
        discounted_returns = []
        for i in range(len(rewards)):
            future_rewards = np.float32(rewards[i:])
            discount_multipliers = discount_multipliers[:len(future_rewards)]
            discounted_return = np.sum(future_rewards * discount_multipliers)
            discounted_returns.append(discounted_return)
        return discounted_returns

    def learn_value(self, global_episode_ts, episode_data):
        """ Perform a step of training for the Monte-Carlo Value-function
        estimator.

        post_step_hook Parameters:
            global_step_ts (int)
            episode_data (Episode)
        """
        self.v_function.update_td([episode_data])

    def learn(self, global_episode_ts, episode_data):
        """Perform one step of learning with a batch of data

        post_step_hook Parameters:
            global_step_ts (int)
            episode_data (Episode)
        """

        all_states, actions = episode_data.observations, episode_data.actions
        rewards = episode_data.rewards

        v_all_states = self.v_function(all_states)
        v_states, v_state_ns = v_all_states[:-1], v_all_states[1:]
        states = all_states[:-1]

        advantage = rewards + self.gamma * v_state_ns - v_states
        with tf.GradientTape() as tape:
            numerical_prefs = self.model(states)
            logprobs = self.logprobs(numerical_prefs, actions)

            average_entropy = tf.reduce_mean(
                self.policy_entropy(numerical_prefs))

            losses = -tf.reduce_sum(logprobs * advantage)
            losses -= self.entropy_coeff * average_entropy

            grads = tape.gradient(losses, self.model.trainable_weights)
            if (global_episode_ts - 1) % self.n_workers == 0:
                if self.accumulated_grads is not None:
                    self.opt.apply_gradients(
                        zip(self.accumulated_grads,
                            self.model.trainable_weights))
                self.accumulated_grads = grads
            else:
                self.accumulated_grads = [
                    ag + g for g, ag in zip(grads, self.accumulated_grads)
                ]

        self.stats.append("episode_average_entropy", global_episode_ts,
                          average_entropy)
        self.stats.append("episode_average_advantage", global_episode_ts,
                          np.mean(advantage))

        self.stats.append("episode_losses", global_episode_ts, losses)


class A2CContinuousAgent(GaussianPolicyMX, A2CAgent):
    """A2C agent for continuous actions using a gaussian
    policy

    See examples/a2c.py for example agent
    """

    def __init__(self,
                 env,
                 model,
                 policy_learning_rate,
                 v_function,
                 gamma=0.9,
                 entropy_coeff=3,
                 n_workers=2):
        """
        Parameters:
        model: A callable model with the final layer being identity.
        v_function: A callable model of the state-value function
        """
        BaseAgent.__init__(self, env)
        GaussianPolicyMX.__init__(self)

        self.gamma = gamma
        self.n_workers = n_workers
        self.entropy_coeff = entropy_coeff

        self.model = model
        self.opt = tf.train.AdamOptimizer(policy_learning_rate)
        self.post_episode_hooks.append(self.learn)

        self.v_function = v_function
        self.post_episode_hooks.append(self.learn_value)

        self.accumulated_grads = None

        self.model_list = [self.model, self.v_function]
