import os
import random
import numpy as np

from sacred import Experiment
from sacred.observers import MongoObserver

from rlforge.environments.environment import GymEnv
from rlforge.common.value_functions import VNetworkDense
from rlforge.common.policy_functions import PolicyNetworkDense
from rlforge.agents.pg import REINFORCEDiscreteAgent, REINFORCEContinuousAgent


def example_discrete():
    """A simple example of REINFORCE agent with discrete actions
    """
    ex = Experiment()
    # ex.observers.append(MongoObserver.create())

    @ex.config
    def config_discrete():
        env_name = 'CartPole-v0'
        seed = 100
        gamma = 0.9
        activation = "tanh"
        entropy_coeff = 0.0001

        policy_layer_sizes = [64, 64]
        policy_learning_rate = 5e-4

        baseline_layer_sizes = [64, 64]
        baseline_learning_rate = 1e-2

        n_train_episodes = 250

    @ex.automain
    def train_example(env_name, seed, gamma, policy_layer_sizes,
                      baseline_layer_sizes, policy_learning_rate,
                      entropy_coeff, baseline_learning_rate,
                      activation, n_train_episodes):
        env = GymEnv(env_name)

        random.seed(seed)
        np.random.seed(seed)
        env.env.seed(seed)
        os.environ['CHAINER_SEED'] = str(seed)

        policy_config = dict(layer_sizes=policy_layer_sizes,
                             activation=activation)
        policy = PolicyNetworkDense(env.n_actions, policy_config)

        # value_config = dict(layer_sizes=baseline_layer_sizes,
                            # activation=activation)
        # value_opt = tf.train.AdamOptimizer(baseline_learning_rate)
        # value_baseline = VNetworkDense(value_config, gamma, value_opt)
        value_baseline = None

        agent = REINFORCEDiscreteAgent(
                               env=env, 
                               model=policy,
                               policy_learning_rate=policy_learning_rate,
                               baseline=value_baseline,
                               gamma=gamma,
                               entropy_coeff=entropy_coeff,
                               experiment=ex)

        agent.interact(n_train_episodes, show_progress=True)


def example_continuous():
    ex = Experiment()
    ex.observers.append(MongoObserver.create())

    @ex.config
    def config_continuous():
        env_name = 'MountainCarContinuous-v0'
        seed = 100
        gamma = 0.7
        activation = "tanh"
        entropy_coeff = 0.001

        policy_layer_sizes = [64, 64]
        policy_learning_rate = 5e-4

        baseline_layer_sizes = [64, 64]
        baseline_learning_rate = 5e-4

        n_train_episodes = 250

    @ex.automain
    def train_example(env_name, seed, gamma, policy_layer_sizes,
                      baseline_layer_sizes, policy_learning_rate,
                      entropy_coeff, baseline_learning_rate,
                      activation, n_train_episodes):
        env = GymEnv(env_name)

        # Set the random seeds for reproducability.
        random.seed(seed)
        np.random.seed(seed)
        env.env.seed(seed)
        os.environ['CHAINER_SEED'] = str(seed)


        policy_config = dict(layer_sizes=policy_layer_sizes,
                             activation=activation)
        policy = PolicyNetworkDense(2 * env.n_actions, policy_config)

        # value_config = dict(layer_sizes=baseline_layer_sizes,
        #                     activation=activation)
        # value_opt = tf.train.AdamOptimizer(baseline_learning_rate)
        # value_baseline = VNetworkDense(value_config, gamma, value_opt)
        value_baseline=None
        agent = REINFORCEContinuousAgent(env=env,
                                         model=policy,
                                         policy_learning_rate=policy_learning_rate,
                                         baseline=value_baseline,
                                         gamma=gamma,
                                         entropy_coeff=entropy_coeff)

        agent.interact(n_train_episodes, show_progress=True)


if __name__ == "__main__":
    example_continuous()
    # example_discrete()
