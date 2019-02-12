import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

from tqdm import tqdm

from rlforge.environments.environment import GymEnv
from rlforge.runners.trainers import train_sequential
# from rlforge.common.policy_functions import ValuePolicyNetworkDense
from rlforge.common.value_functions import ValuePolicyNetworkDense
from rlforge.agents.a2c import A2CAgent, A2CContinuousAgent


def example_discrete():
    env = GymEnv('CartPole-v0')

    np.random.seed(0)
    tf.set_random_seed(0)
    env.env.seed(0)

    gamma = 0.9
    model_learning_rate = 0.0005
    v_function_coeff = 5

    network_config = dict(layer_sizes=[512, 512], activation="tanh")
    output_sizes = [env.n_actions, 1]
    model = ValuePolicyNetworkDense(
        network_config, output_sizes, gamma, n_steps=8)

    agent = A2CAgent(
        env,
        model,
        model_learning_rate,
        v_function_coeff=v_function_coeff,
        gamma=gamma,
        entropy_coeff=3.5,
        n_workers=6)
    train_sequential(agent, env, 250, seed=list(range(5)))


def example_continuous():
    env = GymEnv("MountainCarContinuous-v0")

    np.random.seed(0)
    tf.set_random_seed(0)
    env.env.seed(0)

    gamma = 0.9
    model_learning_rate = 0.0005
    v_function_coeff = 0.01

    network_config = dict(layer_sizes=[512, 512], activation="tanh")
    output_sizes = [2 * env.n_actions, 1]
    model = ValuePolicyNetworkDense(network_config, output_sizes, n_steps=3)

    agent = A2CContinuousAgent(
        env,
        model,
        model_learning_rate,
        v_function_coeff=v_function_coeff,
        gamma=gamma,
        entropy_coeff=2.5,
        n_workers=6)
    train_sequential(agent, env, 250, seed=list(range(5)))


if __name__ == "__main__":
    # example_continuous()
    example_discrete()