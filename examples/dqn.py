import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

from tqdm import tqdm

from rlforge.agents.dqn import DQNAgent
from rlforge.common.value_functions import QNetworkDense
from rlforge.environments.environment import GymEnv

def train(agent, n_episodes):
    # Simple train function using tqdm to show progress
    pbar = tqdm(range(n_episodes))
    for i in pbar:
        agent.interact(1)
        if i % 5 == 0:
            last_5_rets = agent.stats.get_values("episode_returns")[-5:]
            pbar.set_description("Latest return: " +
                                 str(np.mean(last_5_rets)))



if __name__ == '__main__':
    env = GymEnv('CartPole-v0')

    np.random.seed(0)
    tf.set_random_seed(0)
    env.env.seed(0)

    policy_learning_rate = 0.0001
    gamma = 0.8

    q_config = dict(layer_sizes=[64, 64], activation="tanh")
    q_opt = tf.train.AdamOptimizer(policy_learning_rate)
    q_network = QNetworkDense(q_config, q_opt, gamma)

    agent = DQNAgent(env, q_network, policy_learning_rate=policy_learning_rate,
                     replay_buffer_size=10000,
                     target_network_update_freq=300,
                     gamma=0.8,
                     eps=0.2,
                     minibatch_size=128)
    train(agent, 250)
    print("Average Return (Train)", np.mean(
        agent.stats.get_values("episode_returns")))