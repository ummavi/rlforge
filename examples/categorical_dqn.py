import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

from tqdm import tqdm

from rlforge.agents.categorical_dqn import CategoricalDQN
from rlforge.common.value_functions import QNetworkDense
from rlforge.environments.environment import GymEnv
from rlforge.common.networks import DenseBlock, Sequential


def ParametericNetworkDense(n_actions, n_atoms, network_config):
    final_layer_config = dict(network_config)
    final_layer_config.update(dict(layer_sizes=[n_atoms * n_actions],
                                   activation="linear"))
    return Sequential([DenseBlock(params=network_config),
                       DenseBlock(params=final_layer_config),
                       tf.keras.layers.Reshape([n_actions, n_atoms])])



def train(agent, n_episodes):
    # Simple train function using tqdm to show progress
    pbar = tqdm(range(n_episodes))
    for i in pbar:
        e = agent.interact(1)
        if i % 5 == 0:
            last_5_rets = agent.stats.get_values("episode_returns")[-5:]
            pbar.set_description("Latest return: " +
                                 str(np.mean(last_5_rets)))

if __name__=="__main__":
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
