import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

from tqdm import tqdm

from rlforge.environments.environment import GymEnv
from rlforge.runners.trainers import train_sequential
# from rlforge.common.policy_functions import ValuePolicyNetworkDense
from rlforge.common.value_functions import ValuePolicyNetworkDense
from rlforge.agents.a2c import A2CAgent
from rlforge.modules.sil import SILMX


class A2CSILAgent(SILMX, A2CAgent):
    def __init__(self, env, model, model_learning_rate, a2c_value_coeff, gamma,
                 entropy_coeff, n_workers, sil_buffer_size, sil_minibatch_size,
                 sil_n_train_steps, sil_value_coeff):

        A2CAgent.__init__(self, env, model, model_learning_rate,
                          a2c_value_coeff, gamma, entropy_coeff, n_workers)

        SILMX.__init__(self, sil_buffer_size, sil_minibatch_size,
                       sil_n_train_steps, sil_value_coeff, sil_run_every=n_workers)


def example_discrete():
    env = GymEnv('LunarLander-v2')

    np.random.seed(0)
    tf.set_random_seed(0)
    env.env.seed(0)

    gamma = 0.9
    model_learning_rate = 0.00025
    a2c_value_coeff = 5

    sil_value_coeff = 2.5
    sil_buffer_size = 500
    sil_minibatch_size = 16
    sil_n_train_steps = 1

    network_config = dict(layer_sizes=[512, 512], activation="tanh")
    output_sizes = [env.n_actions, 1]
    model = ValuePolicyNetworkDense(
        network_config, output_sizes, gamma, n_steps=8)

    agent = A2CSILAgent(
        env,
        model,
        model_learning_rate,
        a2c_value_coeff,
        gamma,
        entropy_coeff=3.5,
        n_workers=6,
        sil_buffer_size=sil_buffer_size,
        sil_minibatch_size=sil_minibatch_size,
        sil_n_train_steps=sil_n_train_steps,
        sil_value_coeff=sil_value_coeff)
    train_sequential(agent, env, 250, seed=list(range(5)))


if __name__ == "__main__":
    import time
    print("Started at time",time.time())
    # example_continuous()
    example_discrete()