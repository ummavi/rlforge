import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

from rlforge.agents.base_agent import BaseAgent
from rlforge.mixins.policies import SoftmaxPolicyMX, GaussianPolicyMX
from rlforge.common.utils import discounted_returns


class REINFORCEAgent(SoftmaxPolicyMX, BaseAgent):
    """
    Simple Episodic REINFORCE (Williams92) agent with discrete actions
    and a softmax policy.

    See examples/reinforce.py for example agent
    """

    def __init__(self,
                 env,
                 model,
                 policy_learning_rate,
                 baseline=None,
                 baseline_learning_rate=None,
                 gamma=0.9,
                 entropy_coeff=0.0):
        """
        Parameters:
        model: A callable model with the final layer being identity.
        baseline (None/Value Function): Optional baseline function
        """
        BaseAgent.__init__(self, env)
        SoftmaxPolicyMX.__init__(self)

        assert env.action_space == "discrete"

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
            self.post_episode_hooks.append(self.learn_baseline)

        self.model_list = [self.model, self.baseline]

    def learn_baseline(self, global_episode_ts, episode_data):
        """ Perform a step of training for the Monte-Carlo Value-function
        estimator.

        post_step_hook Parameters:
            global_step_ts (int)
            episode_data (Episode)
        """
        self.baseline.update_mc([episode_data])
        # self.baseline.update_td([episode_data])

    def learn(self, global_episode_ts, episode_data):
        """Perform one step of learning with a batch of data

        post_step_hook Parameters:
            global_step_ts (int)
            episode_data (Episode)
        """

        returns = discounted_returns(episode_data.rewards, gamma=self.gamma)
        states, actions = episode_data.observations[:-1], episode_data.actions

        returns = returns - self.baseline(states)
        with tf.GradientTape() as tape:
            # Obtain the numerical preferences from the model.
            # .. Depending on the policy, this could be output to
            # .. the softmax or the parameters of the gaussian
            # .. the name's from Sutton & Barto 2nd ed.
            numerical_prefs = self.model(states)
            logprobs = self.logprobs(numerical_prefs, actions)

            average_entropy = tf.reduce_mean(
                self.policy_entropy(numerical_prefs))

            losses = -tf.reduce_sum(logprobs * returns)
            losses -= self.entropy_coeff * average_entropy

            grads = tape.gradient(losses, self.model.trainable_weights)
            self.opt.apply_gradients(zip(grads, self.model.trainable_weights))

        # self.stats.append("episode_average_entropy", global_episode_ts,
        # average_entropy)
        self.stats.append("episode_losses", global_episode_ts, losses)


class REINFORCEContinuousAgent(GaussianPolicyMX, REINFORCEAgent):
    """
    Simple Episodic REINFORCE (Williams92) agent with discrete actions
    and a softmax policy.

    See examples/reinforce.py for example agent
    """

    def __init__(self,
                 env,
                 model,
                 policy_learning_rate,
                 baseline=None,
                 baseline_learning_rate=None,
                 gamma=0.9,
                 entropy_coeff=0.0):
        """
        Parameters:
        model: A callable model with the final layer being identity.
        baseline (None/Value Function): Optional baseline function
        """
        BaseAgent.__init__(self, env)
        GaussianPolicyMX.__init__(self)

        assert env.action_space == "continuous"
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
            self.post_episode_hooks.append(self.learn_baseline)

        self.model_list = [self.model, self.baseline]
