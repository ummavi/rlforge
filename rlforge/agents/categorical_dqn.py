import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

from rlforge.agents.base_agent import BaseAgent
from rlforge.mixins.policies import DistributionalPolicyMX
from rlforge.mixins.experience_replay import ExperienceReplayMX
from rlforge.mixins.target_network import TargetNetworkMX


class CategoricalDQN(DistributionalPolicyMX, ExperienceReplayMX, BaseAgent):
    """Categorical/Distributional DQN agent

    Implementation of the categorical DQN  paper described in
    (https://arxiv.org/abs/1707.06887)

    TODO:
    * Optimize projection code. Handle the edge cases
    * Use target network
    """

    def __init__(self, env, q_function, policy_learning_rate,
                 replay_buffer_size,
                 minibatch_size=32, gamma=0.98, ts_start_learning=200,
                 eps=0.2, eps_schedule=None, eps_start=None,
                 eps_end=None, ts_eps_end=None,
                 n_atoms=51, v_min=0, v_max=200):

        self.model = q_function
        self.gamma = gamma
        self.ts_start_learning = ts_start_learning

        BaseAgent.__init__(self, env)
        ExperienceReplayMX.__init__(self, replay_buffer_size, minibatch_size)
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

    def categorical_projection(self, state_ns, rewards):
        """
        Applies the categorical projection listed under algorithm 1.
        """
        batch_size = len(state_ns)
        p = self.atom_probabilities(state_ns)
        q_s_n = np.dot(p, np.transpose(self.z))
        a_star = tf.argmax(q_s_n, axis=-1)
        a_mask = tf.expand_dims(tf.one_hot(a_star, self.env.n_actions), axis=2)
        p_astar = tf.reduce_sum(p * a_mask, axis=1)

        m = np.zeros((batch_size, self.n_atoms), dtype=np.float32)

        z_n = tf.tile(tf.expand_dims(rewards, axis=1), [
                      1, self.n_atoms]) + self.gamma * self.z
        z_n = np.clip(z_n, self.v_min, self.v_max)

        b = (z_n - self.v_min) / self.delta_z

        l, u = np.int32(np.floor(b)), np.int32(np.ceil(b))

        for i in range(batch_size):
            for j in range(self.n_atoms):
                l_index, u_index = l[i][j], u[i][j]
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

        # if global_step_ts < self.ts_start_learning:
        # return

        states, actions, rewards, state_ns, dones = self.get_train_batch()
        rewards, is_not_terminal = np.float32(
            rewards), np.float32(np.invert(dones))

        m = self.categorical_projection(state_ns, rewards)

        with tf.GradientTape() as tape:
            p = self.model(states)
            a_mask = tf.expand_dims(tf.one_hot(
                actions, self.env.n_actions), axis=2)
            p_selected = tf.reduce_sum(p * a_mask, axis=1)

            losses = tf.reduce_mean(-tf.reduce_sum(m *
                                                   tf.log(p_selected), axis=-1))

            grads = tape.gradient(losses, self.model.trainable_weights)
            self.opt.apply_gradients(zip(grads,
                                         self.model.trainable_weights))

        # self.stats.append("step_losses", global_step_ts, losses)
        # self.stats.append("step_mean_q", global_step_ts, np.mean(preds))


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
            # if i % 5 == 0:
            last_5_rets = agent.stats.get_values("episode_returns")[-1:]
            pbar.set_description("Latest return: " +
                                 str(np.mean(last_5_rets)))

    env = GymEnv('CartPole-v0')

    np.random.seed(0)
    tf.set_random_seed(0)
    env.env.seed(0)

    q_network = ParametericNetworkDense(env.n_actions, 51, dict(layer_sizes=[64, 64],
                                                                activation="tanh"))
    agent = CategoricalDQN(env, q_network, policy_learning_rate=0.001,
                           replay_buffer_size=10000,
                           gamma=0.8,
                           eps=0.2,
                           minibatch_size=32)
    train(agent, 250)
    print("Average Return (Train)", np.mean(
        agent.stats.get_values("episode_returns")))
