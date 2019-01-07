import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

from rlforge.agents.base_agent import BaseAgent
from rlforge.mixins.policies import DistributionalPolicyMX
from rlforge.mixins.experience_replay import ExperienceReplayMX
from rlforge.mixins.target_network import TargetNetworkMX
from rlforge.common.utils import one_hot


class CategoricalDQN(DistributionalPolicyMX, ExperienceReplayMX, TargetNetworkMX, BaseAgent):
    """Categorical/Distributional DQN agent

    Implementation of the categorical DQN  paper described in
    (https://arxiv.org/abs/1707.06887)

    TODO:
    * Optimize projection code.
    """

    def __init__(self, env, q_function, policy_learning_rate,
                 replay_buffer_size, target_network_update_freq,
                 minibatch_size=32, gamma=0.98, ts_start_learning=200,
                 eps=0.2, eps_schedule=None, eps_start=None,
                 eps_end=None, ts_eps_end=None,
                 n_atoms=51, v_min=0, v_max=200):

        self.model = q_function
        self.gamma = gamma
        self.ts_start_learning = ts_start_learning

        BaseAgent.__init__(self, env)
        ExperienceReplayMX.__init__(self, replay_buffer_size, minibatch_size)
        TargetNetworkMX.__init__(self, target_network_update_freq)
        DistributionalPolicyMX.__init__(self, eps_fixed=eps,
                                        eps_schedule=eps_schedule,
                                        eps_start=eps_start, eps_end=eps_end,
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
        is_not_terminal = np.tile(np.expand_dims(
            is_not_terminal, axis=1), [1, self.n_atoms])

        z_n = rewards + self.gamma * self.z * is_not_terminal
        z_n = np.clip(z_n, self.v_min, self.v_max)

        b = (z_n - self.v_min) / self.delta_z
        l, u = np.floor(b), np.ceil(b)

        for i in range(batch_size):
            for j in range(self.n_atoms):
                l_index, u_index = int(l[i][j]), int(u[i][j])
                m[i][l_index] += np.float32(p_astar[i]
                                            [j] * (u_index - b[i][j]))
                m[i][u_index] += np.float32(p_astar[i]
                                            [j] * (b[i][j] - l_index))
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
        rewards, is_not_terminal = np.float32(
            rewards), np.float32(np.invert(dones))

        m = self.categorical_projection(state_ns, rewards, is_not_terminal)

        with tf.GradientTape() as tape:
            logp = tf.nn.log_softmax(self.model(states))
            a_mask = tf.expand_dims(tf.one_hot(
                actions, self.env.n_actions), axis=2)
            a_mask = tf.tile(a_mask,[1,1,self.n_atoms])
            logp_selected = tf.reduce_sum(logp * a_mask, axis=1)
            losses = tf.reduce_mean(-tf.reduce_sum(m * logp_selected, axis=-1))

            grads = tape.gradient(losses, self.model.trainable_weights)
            self.opt.apply_gradients(zip(grads,
                                         self.model.trainable_weights))


from rlforge.common.networks import DenseBlock, Sequential
def ParametericNetworkDense(n_actions, n_atoms, network_config):
    final_layer_config = dict(network_config)
    final_layer_config.update(dict(layer_sizes=[n_atoms * n_actions],
                                   activation="linear"))
    return Sequential([DenseBlock(params=network_config),
                       DenseBlock(params=final_layer_config),
                       tf.keras.layers.Reshape([n_actions, n_atoms])])


if __name__ == '__main__':
    from tqdm import tqdm
    from rlforge.common.value_functions import QNetworkDense
    from rlforge.environments.environment import GymEnv

    def train(agent, n_episodes):
        # Simple train function using tqdm to show progress
        pbar = tqdm(range(n_episodes))
        for i in pbar:
            e = agent.interact(1)
            if i % 5 == 0:
                last_5_rets = agent.stats.get_values("episode_returns")[-5:]
                pbar.set_description("Latest return: " +
                                     str(np.mean(last_5_rets)))

    n_atoms = 51
    env = GymEnv('CartPole-v0')

    np.random.seed(0)
    tf.set_random_seed(0)
    env.env.seed(0)

    q_network = ParametericNetworkDense(env.n_actions, n_atoms, dict(layer_sizes=[64, 64],
                                                                     activation="tanh"))
    agent = CategoricalDQN(env, q_network, policy_learning_rate=0.005,
                           replay_buffer_size=10000, target_network_update_freq=200,
                           gamma=0.8,
                           eps=0.2,
                           minibatch_size=128,
                           n_atoms=n_atoms)
    train(agent, 100)
    print("Average Return (Train)", np.mean(
        agent.stats.get_values("episode_returns")))
