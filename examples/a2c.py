import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

from tqdm import tqdm

from rlforge.environments.environment import GymEnv
from rlforge.runners.trainers import train_sequential
from rlforge.common.value_functions import VNetworkDense
from rlforge.common.policy_functions import PolicyNetworkDense
from rlforge.agents.a2c import A2CAgent, A2CContinuousAgent


def example_discrete():
    """A simple example of REINFORCE agent with discrete actions
    """
    env = GymEnv('CartPole-v0')

    np.random.seed(0)
    tf.set_random_seed(0)
    env.env.seed(0)

    gamma = 0.9

    actor_learning_rate = 0.0005
    policy_config = dict(layer_sizes=[64, 64], activation="tanh")
    policy = PolicyNetworkDense(env.n_actions, policy_config)

    baseline_learning_rate = 0.0005
    value_config = dict(layer_sizes=[64, 64], activation="tanh")
    value_opt = tf.train.AdamOptimizer(baseline_learning_rate)
    value_critic = VNetworkDense(value_config, value_opt, gamma)
    agent = A2CAgent(
        env,
        policy,
        policy_learning_rate=actor_learning_rate,
        v_function=value_critic,
        gamma=gamma,
        entropy_coeff=0.0001)
    train_sequential(agent, env, 250, seed=list(range(5)))


def example_continuous():
    env = GymEnv("MountainCarContinuous-v0")

    np.random.seed(0)
    tf.set_random_seed(0)
    env.env.seed(0)

    gamma = 0.9

    actor_learning_rate = 0.005
    policy_config = dict(layer_sizes=[64, 64], activation="tanh")
    policy = PolicyNetworkDense(2 * env.n_actions, policy_config)

    critic_learning_rate = 0.001
    value_config = dict(layer_sizes=[64, 64], activation="tanh")
    value_opt = tf.train.AdamOptimizer(critic_learning_rate)
    value_critic = VNetworkDense(value_config, value_opt, gamma)

    agent = A2CContinuousAgent(
        env,
        policy,
        v_function=value_critic,
        policy_learning_rate=actor_learning_rate,
        gamma=gamma,
        entropy_coeff=0.01)
    train_sequential(agent, env, 250, seed=list(range(5)))


if __name__ == "__main__":
    example_continuous()
    # example_discrete()