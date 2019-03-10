import os
import random
import chainer
import numpy as np

import chainer.functions as F
from rlforge.environments.environment import GymEnv
from rlforge.agents.categorical_dqn import CategoricalDQN

from rlforge.networks.blocks import DenseBlock

from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment()
ex.observers.append(MongoObserver.create())


class ParametericNetworkDense(chainer.Chain):
    def __init__(self, n_actions, n_atoms, network_config):
        super().__init__()

        self.n_actions = n_actions
        self.n_atoms = n_atoms

        final_layer_config = dict(network_config)
        final_layer_config.update(
            dict(layer_sizes=[n_atoms * n_actions], activation="identity"))
        blocks = [
            DenseBlock(params=network_config),
            DenseBlock(final_layer_config),
        ]
        self.blocks = chainer.ChainList(*blocks)
        for block in self.blocks:
            block.build_block()

    def forward(self, x):
        """Performs a forward pass of the network.
        __call__() automatically redirects to forward
        """
        x = np.float32(x)
        for block in self.blocks:
            x = block(x)
        x = F.reshape(x, [-1, self.n_actions, self.n_atoms])
        return x

    def clone(self):
        return self.copy("copy")


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

    model_layer_sizes = [64, 64]
    model_learning_rate = 5e-4

    replay_buffer_size = 10000
    target_network_update_freq = 200


@ex.automain
def train_example(env_name, seed, gamma, n_atoms, model_layer_sizes,
                  activation, model_learning_rate, replay_buffer_size,
                  target_network_update_freq, minibatch_size, eps,
                  n_train_episodes):
    env = GymEnv(env_name)

    random.seed(seed)
    np.random.seed(seed)
    env.env.seed(seed)
    os.environ['CHAINER_SEED'] = str(seed)

    q_network = ParametericNetworkDense(
        env.n_actions, n_atoms,
        dict(layer_sizes=model_layer_sizes, activation=activation))
    agent = CategoricalDQN(
        env,
        q_network,
        model_learning_rate=model_learning_rate,
        replay_buffer_size=replay_buffer_size,
        target_network_update_freq=target_network_update_freq,
        gamma=gamma,
        eps=eps,
        minibatch_size=minibatch_size,
        n_atoms=n_atoms)

    agent.interact(n_train_episodes, show_progress=True)
