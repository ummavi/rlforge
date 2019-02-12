import numpy as np
import tensorflow as tf

from rlforge.agents.base_agent import BaseAgent
from rlforge.modules.policies import SoftmaxPolicyMX, GaussianPolicyMX


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
    """

    def __init__(self,
                 env,
                 model,
                 model_learning_rate,
                 v_function_coeff,
                 gamma=0.9,
                 entropy_coeff=3,
                 n_workers=2):
        """
        Parameters:
        model: A callable model with the final layer being identity.
        v_function: A callable model of the state-value function
        v_function_coeff: Coeff. the V-function loss should be scaled by
        """
        BaseAgent.__init__(self, env)
        SoftmaxPolicyMX.__init__(self)

        self.gamma = gamma
        self.model = model
        self.n_workers = n_workers
        self.entropy_coeff = entropy_coeff

        self.opt = tf.train.AdamOptimizer(model_learning_rate)

        self.post_episode_hooks.append(self.learn)

        self.v_function_coeff = v_function_coeff

        self.acc_grads = None

        self.model_list = [self.model]

    def learn(self, global_episode_ts, episode_data):
        """Perform one step of learning with a batch of data

        post_step_hook Parameters:
            global_step_ts (int)
            episode_data (Episode)
        """

        states, actions = episode_data.observations, episode_data.actions
        q_sts = self.model.nstep_bootstrapped_value(episode_data)
        # Terminal state's value is 0.
        q_sts = np.squeeze(np.float32(np.concatenate((q_sts, [0.0]))))

        with tf.GradientTape() as tape:
            numerical_prefs, v_sts = self.model(states)
            numerical_prefs = numerical_prefs[:-1]  # Ignore A_{T}
            v_sts = tf.squeeze(v_sts)

            advantage = q_sts - v_sts
            advantage = tf.stop_gradient(advantage)

            # Policy Loss
            logprobs = self.logprobs(numerical_prefs, actions)
            average_entropy = tf.reduce_mean(
                self.policy_entropy(numerical_prefs))

            loss_policy = -tf.reduce_sum(logprobs * advantage[:-1])
            loss_policy -= self.entropy_coeff * average_entropy

            # Value loss
            loss_value = tf.losses.mean_squared_error(q_sts, v_sts)
            losses = loss_policy + self.v_function_coeff * loss_value

            grads = tape.gradient(losses, self.model.trainable_weights)
            # grads, _ = tf.clip_by_global_norm(grads, 40.0)

            if self.acc_grads is None:
                self.acc_grads = grads
            else:
                self.acc_grads = [
                    ag + g for g, ag in zip(grads, self.acc_grads)
                ]

            if (global_episode_ts) % self.n_workers == 0:
                self.opt.apply_gradients(
                    zip(self.acc_grads, self.model.trainable_weights))
                self.acc_grads = None

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
                 model_learning_rate,
                 v_function_coeff,
                 gamma=0.9,
                 entropy_coeff=3,
                 n_workers=2):
        """
        Parameters:
        model: A callable model with the final layer being identity.
        v_function: A callable model of the state-value function
        v_function_coeff: Coeff. the V-function loss should be scaled by
        """
        BaseAgent.__init__(self, env)
        GaussianPolicyMX.__init__(self)

        self.gamma = gamma
        self.n_workers = n_workers
        self.entropy_coeff = entropy_coeff

        self.model = model
        self.opt = tf.train.AdamOptimizer(model_learning_rate)
        self.post_episode_hooks.append(self.learn)

        self.v_function_coeff = v_function_coeff

        self.acc_grads = None

        self.model_list = [self.model]
