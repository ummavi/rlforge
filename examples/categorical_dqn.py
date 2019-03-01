import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

from rlforge.environments.environment import GymEnv
from rlforge.agents.categorical_dqn import CategoricalDQN
from rlforge.common.networks import DenseBlock, Sequential

from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment()
ex.observers.append(MongoObserver.create())


def ParametericNetworkDense(n_actions, n_atoms, network_config):
    final_layer_config = dict(network_config)
    final_layer_config.update(
        dict(layer_sizes=[n_atoms * n_actions], activation="linear"))
    return Sequential([
        DenseBlock(params=network_config),
        DenseBlock(params=final_layer_config),
        tf.keras.layers.Reshape([n_actions, n_atoms])
    ])


@ex.config
def config():
    env_name = 'CartPole-v0'
    eps = 0.2
    seed = 100
    gamma = 0.9
    n_atoms = 51
    activation = "tanh"
    minibatch_size = 128
    n_train_episodes = 250

    policy_layer_sizes = [64, 64]
    policy_learning_rate = 5e-4

    replay_buffer_size = 10000
    target_network_update_freq = 200


@ex.automain
def train_example(env_name, seed, gamma, n_atoms, policy_layer_sizes,
                  activation, policy_learning_rate, replay_buffer_size,
                  target_network_update_freq, minibatch_size, eps,
                  n_train_episodes):
    env = GymEnv(env_name)

    np.random.seed(seed)
    tf.set_random_seed(seed)
    env.env.seed(seed)

    q_network = ParametericNetworkDense(
        env.n_actions, n_atoms,
        dict(layer_sizes=policy_layer_sizes, activation=activation))
    agent = CategoricalDQN(
        env,
        q_network,
        policy_learning_rate=policy_learning_rate,
        replay_buffer_size=replay_buffer_size,
        target_network_update_freq=target_network_update_freq,
        gamma=gamma,
        eps=eps,
        minibatch_size=minibatch_size,
        n_atoms=n_atoms)

    agent.interact(n_train_episodes, show_progress=True)
