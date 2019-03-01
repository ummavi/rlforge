import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

from rlforge.environments.environment import GymEnv
from rlforge.common.value_functions import ValuePolicyNetworkDense
from rlforge.agents.a2c import A2CAgent
from rlforge.modules.sil import SILMX


class A2CSILAgent(SILMX, A2CAgent):
    def __init__(self, env, model, model_learning_rate, a2c_value_coeff, gamma,
                 entropy_coeff, n_workers, sil_buffer_size, sil_minibatch_size,
                 sil_n_train_steps, sil_value_coeff):

        A2CAgent.__init__(self, env, model, model_learning_rate,
                          a2c_value_coeff, gamma, entropy_coeff, n_workers)

        SILMX.__init__(
            self,
            sil_buffer_size,
            sil_minibatch_size,
            sil_n_train_steps,
            sil_value_coeff,
            sil_run_every=n_workers)


from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment()
ex.observers.append(MongoObserver.create())


@ex.config
def config():
    env_name = 'LunarLander-v2'
    seed = 100
    gamma = 0.9
    activation = "tanh"

    model_layer_sizes = [512, 512]
    model_learning_rate = 0.00025

    n_train_episodes = 250
    a2c_value_coeff = 5

    sil_value_coeff = 2.5
    sil_buffer_size = 500
    sil_minibatch_size = 16
    sil_n_train_steps = 1

    n_steps = 8
    entropy_coeff = 3.5
    n_workers = 6


@ex.automain
def train_example(env_name, seed, gamma, activation, model_layer_sizes,
                  model_learning_rate, a2c_value_coeff, sil_value_coeff,
                  sil_buffer_size, sil_minibatch_size, sil_n_train_steps,
                  n_steps, entropy_coeff, n_workers, n_train_episodes):
    env = GymEnv(env_name)

    np.random.seed(seed)
    tf.set_random_seed(seed)
    env.env.seed(seed)

    network_config = dict(layer_sizes=model_layer_sizes, activation=activation)
    output_sizes = [env.n_actions, 1]
    model = ValuePolicyNetworkDense(
        network_config, output_sizes, gamma, n_steps=n_steps)

    agent = A2CSILAgent(
        env,
        model,
        model_learning_rate,
        a2c_value_coeff,
        gamma,
        entropy_coeff=entropy_coeff,
        n_workers=n_workers,
        sil_buffer_size=sil_buffer_size,
        sil_minibatch_size=sil_minibatch_size,
        sil_n_train_steps=sil_n_train_steps,
        sil_value_coeff=sil_value_coeff)
    agent.interact(n_train_episodes, show_progress=True)