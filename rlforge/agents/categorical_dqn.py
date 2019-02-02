import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

from rlforge.agents.base_agent import BaseAgent
from rlforge.mixins.policies import DistributionalPolicyMX
from rlforge.mixins.experience_replay import ExperienceReplayMX
from rlforge.mixins.target_network import TargetNetworkMX
from rlforge.common.utils import one_hot


class CategoricalDQN(DistributionalPolicyMX, ExperienceReplayMX,
                     TargetNetworkMX, BaseAgent):
    """Categorical/Distributional DQN agent

    Implementation of the categorical DQN  paper described in
    (https://arxiv.org/abs/1707.06887)

    TODO:
    * Optimize projection code.

    See examples/categorical_dqn.py for example agent
    """

    def __init__(self,
                 env,
                 q_function,
                 policy_learning_rate,
                 replay_buffer_size,
                 target_network_update_freq,
                 minibatch_size=32,
                 gamma=0.98,
                 ts_start_learning=200,
                 eps=0.2,
                 eps_schedule=None,
                 eps_start=None,
                 eps_end=None,
                 ts_eps_end=None,
                 n_atoms=51,
                 v_min=0,
                 v_max=200):

        self.model = q_function
        self.gamma = gamma
        self.ts_start_learning = ts_start_learning

        BaseAgent.__init__(self, env)
        ExperienceReplayMX.__init__(self, replay_buffer_size, minibatch_size)
        TargetNetworkMX.__init__(self, target_network_update_freq)
        DistributionalPolicyMX.__init__(
            self,
            eps_fixed=eps,
            eps_schedule=eps_schedule,
            eps_start=eps_start,
            eps_end=eps_end,
            ts_eps_end=ts_eps_end)

        self.n_atoms = n_atoms
        self.v_min, self.v_max = v_min, v_max
        self.delta_z = (v_max - v_min) / float(n_atoms - 1)
        self.z = np.linspace(v_min, v_max, n_atoms)

        self.opt = tf.train.AdamOptimizer(policy_learning_rate)

        # DQN trains after every step so add it to the post_episode hook
        self.post_step_hooks.append(self.learn)

    def categorical_projection(self, state_ns, rewards, is_not_terminal):
        """
        Applies the categorical projection listed under algorithm 1.
        """
        batch_size = len(state_ns)
        # p = self.atom_probabilities(state_ns)
        p = tf.nn.softmax(self.target_model(state_ns))
        self.stats.append("test3", 1, p)

        q_s_n = np.dot(p, np.transpose(self.z))
        assert q_s_n.shape == (batch_size, self.env.n_actions)

        a_star = np.argmax(q_s_n, axis=1)
        a_mask = np.expand_dims(one_hot(a_star, self.env.n_actions), axis=2)
        p_astar = np.sum(p * a_mask, axis=1)

        m = np.zeros((batch_size, self.n_atoms), dtype=np.float32)

        rewards = np.tile(np.expand_dims(rewards, axis=1), [1, self.n_atoms])
        is_not_terminal = np.tile(
            np.expand_dims(is_not_terminal, axis=1), [1, self.n_atoms])

        z_n = rewards + self.gamma * self.z * is_not_terminal
        z_n = np.clip(z_n, self.v_min, self.v_max)

        b = (z_n - self.v_min) / self.delta_z
        l, u = np.floor(b), np.ceil(b)

        for i in range(batch_size):
            for j in range(self.n_atoms):
                l_index, u_index = int(l[i][j]), int(u[i][j])
                m[i][l_index] += np.float32(
                    p_astar[i][j] * (u_index - b[i][j]))
                m[i][u_index] += np.float32(
                    p_astar[i][j] * (b[i][j] - l_index))
        return m

    def learn(self, global_step_ts, step_data):
        """Perform one step of learning with a batch of data

        post_step_hook Parameters:
            global_step_ts (int)
            step_data (tuple): (s,a,r,s',done)
        """

        if global_step_ts < self.ts_start_learning:
            return

        states, actions, rewards, state_ns, dones = self.get_train_batch()
        rewards, is_not_terminal = np.float32(rewards), np.float32(
            np.invert(dones))

        m = self.categorical_projection(state_ns, rewards, is_not_terminal)

        with tf.GradientTape() as tape:
            logp = tf.nn.log_softmax(self.model(states))
            a_mask = tf.expand_dims(
                tf.one_hot(actions, self.env.n_actions), axis=2)
            a_mask = tf.tile(a_mask, [1, 1, self.n_atoms])
            logp_selected = tf.reduce_sum(logp * a_mask, axis=1)
            losses = tf.reduce_mean(-tf.reduce_sum(m * logp_selected, axis=-1))

            grads = tape.gradient(losses, self.model.trainable_weights)
            self.opt.apply_gradients(zip(grads, self.model.trainable_weights))
