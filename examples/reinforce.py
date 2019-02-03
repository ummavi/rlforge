import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

from tqdm import tqdm

from rlforge.agents.pg import REINFORCEAgent, REINFORCEContinuousAgent
from rlforge.common.policy_functions import PolicyNetworkDense
from rlforge.common.value_functions import VNetworkDense
from rlforge.environments.environment import GymEnv


def train(agent, n_episodes):
    # Simple train function using tqdm to show progress
    pbar = tqdm(range(n_episodes))
    for i in pbar:
        agent.interact(1)
        if i % 5 == 0:
            last_5_rets = agent.stats.get_values("episode_returns")[-5:]
            pbar.set_description("Latest return: " + str(np.mean(last_5_rets)))


def example_discrete():
    """A simple example of REINFORCE agent with discrete actions
    """
    env = GymEnv('CartPole-v0')

    np.random.seed(0)
    tf.set_random_seed(0)
    env.env.seed(0)

    gamma = 0.9
    baseline_learning_rate = 0.001

    policy_config = dict(layer_sizes=[64, 64], activation="tanh")
    policy = PolicyNetworkDense(env.n_actions, policy_config)

    value_config = dict(layer_sizes=[64, 64], activation="tanh")
    value_opt = tf.train.AdamOptimizer(baseline_learning_rate)
    value_baseline = VNetworkDense(value_config, value_opt, gamma)

    agent = REINFORCEAgent(
        env,
        policy,
        policy_learning_rate=0.001,
        baseline=value_baseline,
        gamma=gamma,
        entropy_coeff=3)
    train(agent, 500)
    print("Average Return (Train)",
          np.mean(agent.stats.get_values("episode_returns")))


def example_continuous():
    env = GymEnv("MountainCarContinuous-v0")

    np.random.seed(0)
    tf.set_random_seed(0)
    env.env.seed(0)

    gamma = 0.9

    policy_config = dict(layer_sizes=[64, 64], activation="tanh")
    policy = PolicyNetworkDense(2 * env.n_actions, policy_config)

    # baseline_learning_rate = 0.001
    # value_config = dict(layer_sizes=[64, 64], activation="tanh")
    # value_opt = tf.train.AdamOptimizer(baseline_learning_rate)
    # value_baseline = VNetworkDense(value_config, value_opt, gamma)
    value_baseline = None

    agent = REINFORCEContinuousAgent(
        env,
        policy,
        baseline=value_baseline,
        policy_learning_rate=0.0005,
        gamma=gamma,
        entropy_coeff=0.001)
    train(agent, 500)
    print("Average Return (Train)",
          np.mean(agent.stats.get_values("episode_returns")))


if __name__ == "__main__":
    example_continuous()
    # example_discrete()