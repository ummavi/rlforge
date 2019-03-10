import os
import random
import chainer
import numpy as np

from rlforge.agents.double_dqn import DoubleDQNAgent
from rlforge.environments.environment import GymEnv
from rlforge.common.value_functions import QNetworkDense

from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment()
ex.observers.append(MongoObserver.create())


@ex.config
def config():
    env_name = 'CartPole-v0'
    seed = 100
    gamma = 0.9
    activation = "tanh"

    policy_layer_sizes = [64, 64]
    policy_learning_rate = 5e-4

    n_train_episodes = 250
    replay_buffer_size = 10000
    target_network_update_freq = 300
    eps = 0.2
    minibatch_size = 128


@ex.automain
def train_example(env_name, seed, gamma, policy_layer_sizes,
                  policy_learning_rate, activation, replay_buffer_size,
                  target_network_update_freq, eps, minibatch_size,
                  n_train_episodes):
    env = GymEnv(env_name)

    random.seed(seed)
    np.random.seed(seed)
    env.env.seed(seed)
    os.environ['CHAINER_SEED'] = str(seed)

    q_config = dict(layer_sizes=policy_layer_sizes, activation=activation)
    q_opt = chainer.optimizers.RMSpropGraves(policy_learning_rate)
    q_network = QNetworkDense(q_config, env.n_actions, gamma, optimizer=q_opt)

    agent = DoubleDQNAgent(
        env,
        q_network,
        replay_buffer_size=replay_buffer_size,
        target_network_update_freq=target_network_update_freq,
        gamma=gamma,
        eps=eps,
        minibatch_size=minibatch_size,
        experiment=ex)

    agent.interact(n_train_episodes, show_progress=True)
