import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

from sacred import Experiment
from sacred.observers import MongoObserver

from rlforge.environments.environment import GymEnv
from rlforge.agents.a2c import A2CAgent, A2CContinuousAgent
from rlforge.common.value_functions import ValuePolicyNetworkDenseShared
from rlforge.common.value_functions import ValuePolicyNetworkDense


def example_discrete():
    ex = Experiment()
    ex.observers.append(MongoObserver.create())

    @ex.config
    def discrete():
        env_name = "CartPole-v0"
        n_train_episodes = 250
        seeds = [1, 2, 3, 4, 5]

        model_layer_sizes = [256, 256]
        n_steps = 4
        n_workers = 2
        gamma = 0.99
        activation_fn = "tanh"
        entropy_coeff = 1.0
        v_function_coeff = 3
        model_learning_rate = 1e-3

    @ex.automain
    def train_example(env_name, seed, gamma, model_layer_sizes,
                      model_learning_rate, v_function_coeff, entropy_coeff,
                      n_workers, n_train_episodes, activation_fn, n_steps):
        env = GymEnv(env_name)

        np.random.seed(seed)
        tf.set_random_seed(seed)
        env.env.seed(seed)

        network_config = dict(layer_sizes=model_layer_sizes,
                              activation=activation_fn)
        output_sizes = [env.n_actions, 1]

        model = ValuePolicyNetworkDense(network_config,
                                        output_sizes, gamma,
                                        n_steps=n_steps)
        # The shared network does a rather poor job.
        # .. perhaps the environment is much too simple.
        # model = ValuePolicyNetworkDenseShared(network_config,
                                                 # output_sizes, gamma,
                                                 # n_steps=n_steps)

        agent = A2CAgent(env, model, model_learning_rate,
                         v_function_coeff=v_function_coeff,
                         gamma=gamma,
                         entropy_coeff=entropy_coeff,
                         n_workers=n_workers,
                         experiment=ex)

        agent.interact(n_train_episodes, show_progress=True)


def example_continuous():
    ex = Experiment()
    ex.observers.append(MongoObserver.create())

    @ex.config
    def config_continuous():
        env_name = "MountainCarContinuous-v0"
        gamma = 0.99

        model_layer_sizes = [2, 256]
        model_learning_rate = 1e-3

        seed = 100
        n_steps = 4
        n_workers = 2
        activation_fn = "tanh"
        entropy_coeff = 0.01
        n_train_episodes = 150
        v_function_coeff = 0.01

    @ex.automain
    def train_example(env_name, seed, gamma, model_layer_sizes,
                      model_learning_rate, v_function_coeff, entropy_coeff,
                      n_workers, n_train_episodes, activation_fn, n_steps):
        env = GymEnv(env_name)

        np.random.seed(seed)
        tf.set_random_seed(seed)
        env.env.seed(seed)

        network_config = dict(layer_sizes=model_layer_sizes,
                              activation=activation_fn)
        output_sizes = [2 * env.n_actions, 1]
        combined_model = ValuePolicyNetworkDenseShared(network_config,
                                                       output_sizes, gamma,
                                                       n_steps=n_steps)

        agent = A2CContinuousAgent(env, combined_model, model_learning_rate,
                                   v_function_coeff=v_function_coeff,
                                   gamma=gamma,
                                   entropy_coeff=entropy_coeff,
                                   n_workers=n_workers,
                                   experiment=ex)

        agent.interact(n_train_episodes, show_progress=True)


if __name__ == "__main__":
    # example_continuous()
    example_discrete()
