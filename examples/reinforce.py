import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

from tqdm import tqdm

from rlforge.environments.environment import GymEnv
from rlforge.runners.trainers import train_sequential
from rlforge.common.value_functions import VNetworkDense
from rlforge.common.policy_functions import PolicyNetworkDense
from rlforge.agents.pg import REINFORCEAgent, REINFORCEContinuousAgent


def example_discrete():
    """A simple example of REINFORCE agent with discrete actions
    """
    env = GymEnv('CartPole-v0')

    np.random.seed(0)
    tf.set_random_seed(0)
    env.env.seed(0)

    gamma = 0.9

    policy_config = dict(layer_sizes=[64, 64], activation="tanh")
    policy = PolicyNetworkDense(env.n_actions, policy_config)

    baseline_learning_rate = 0.001
    value_config = dict(layer_sizes=[64, 64], activation="tanh")
    value_opt = tf.train.AdamOptimizer(baseline_learning_rate)
    value_baseline = VNetworkDense(value_config, value_opt, gamma)
    agent = REINFORCEAgent(
        env,
        policy,
        policy_learning_rate=0.0001,
        baseline=value_baseline,
        gamma=gamma,
        entropy_coeff=0.0001)
    train_sequential(agent, env, 250, seed=list(range(5)))


def example_continuous():
    env = GymEnv("MountainCarContinuous-v0")

    np.random.seed(0)
    tf.set_random_seed(0)
    env.env.seed(0)

    gamma = 0.9
    policy_config = dict(layer_sizes=[64, 64], activation="tanh")
    policy = PolicyNetworkDense(2 * env.n_actions, policy_config)

    baseline_learning_rate = 0.001
    value_config = dict(layer_sizes=[64, 64], activation="tanh")
    value_opt = tf.train.AdamOptimizer(baseline_learning_rate)
    value_baseline = VNetworkDense(value_config, value_opt, gamma)

    agent = REINFORCEContinuousAgent(
        env,
        policy,
        baseline=value_baseline,
        policy_learning_rate=0.005,
        gamma=gamma,
        entropy_coeff=3)
    train_sequential(agent, env, 250, seed=list(range(5)))


if __name__ == "__main__":
    example_continuous()
    # example_discrete()