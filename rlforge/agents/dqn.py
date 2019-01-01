import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

from rlforge.common.utils import Episode
from rlforge.agents.base_agent import BaseAgent
from rlforge.mixins.policies import EpsilonGreedyPolicyMX
from rlforge.mixins.experience_replay import ExperienceReplayMX
from rlforge.mixins.target_network import TargetNetworkMX


class DQNAgent(EpsilonGreedyPolicyMX, ExperienceReplayMX,
               TargetNetworkMX, BaseAgent):
    """Deep-Q Network agent

    Implementation of the DQN agent as described in the nature paper
    (https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)


    * Huber Loss
    * Epsilon-greedy Policy (EpsilonGreedyPolicyMX)
    * Experience Replay (ExperienceReplayMX)
    * Uses Target Network to compute Q* for the target (TargetNetworkMX)


    Note: Python inheritance from R-L. List BaseAgent at the end
    """

    def __init__(self, env, q_function, policy_learning_rate,
                 replay_buffer_size, target_network_update_freq,
                 minibatch_size=32, gamma=0.98, ts_start_learning=200,
                 eps=0.2, eps_schedule=None, eps_start=None,
                 eps_end=None, ts_eps_end=None):

        self.model = q_function
        self.gamma = gamma
        self.ts_start_learning = ts_start_learning

        BaseAgent.__init__(self, env)
        ExperienceReplayMX.__init__(self, replay_buffer_size, minibatch_size)
        TargetNetworkMX.__init__(self, target_network_update_freq)
        EpsilonGreedyPolicyMX.__init__(self, eps_fixed=eps,
                                       eps_schedule=eps_schedule,
                                       eps_start=eps_start, eps_end=eps_end,
                                       ts_eps_end=ts_eps_end)

        self.opt = tf.train.AdamOptimizer(policy_learning_rate)

        # DQN trains after every step so add it to the post_episode hook
        self.post_step_hooks.append(self.learn)

    def learn(self, global_step_ts, step_data):
        """Perform one step of learning with a batch of data

        post_step_hook Parameters:
            global_step_ts (int)
            step_data (tuple): (s,a,r,s',done)
        """

        if global_step_ts < self.ts_start_learning:
            return

        states, actions, rewards, state_ns, dones = self.get_train_batch()
        rewards, is_not_terminal = np.float32(
            rewards), np.float32(np.invert(dones))

        q_star = tf.reduce_max(self.target_model(state_ns), axis=-1)
        q_target = rewards + (self.gamma * q_star * is_not_terminal)

        with tf.GradientTape() as tape:
            preds = self.model(states)
            q_cur = tf.reduce_sum(
                preds * tf.one_hot(actions, self.env.n_actions), axis=-1)
            losses = tf.losses.huber_loss(labels=q_target, predictions=q_cur)

            grads = tape.gradient(losses, self.model.trainable_weights)
            self.opt.apply_gradients(zip(grads,
                                         self.model.trainable_weights))

        self.stats.append("step_losses", global_step_ts, losses)
        self.stats.append("step_mean_q", global_step_ts, np.mean(preds))


if __name__ == '__main__':
    from tqdm import tqdm
    from rlforge.common.q_functions import QNetworkDense
    from rlforge.environments.environment import GymEnv

    def train(agent, n_episodes):
        # Simple train function using tqdm to show progress
        pbar = tqdm(range(n_episodes))
        for i in pbar:
            e = agent.interact(1)
            if i % 5 == 0:
                last_5_rets = agent.stats.get_values("episode_returns")[-5:]
                pbar.set_description("Latest return: " +
                                     str(np.mean(last_5_rets)))

    env = GymEnv('CartPole-v0')

    np.random.seed(0)
    tf.set_random_seed(0)
    env.env.seed(0)

    q_network = QNetworkDense(env.n_actions, dict(layer_sizes=[64, 64],
                                                  activation="tanh"))
    agent = DQNAgent(env, q_network, policy_learning_rate=0.0001,
                     target_network_update_freq=300,
                     replay_buffer_size=10000,
                     gamma=0.8,
                     eps=0.2,
                     minibatch_size=128)
    train(agent, 250)
    print("Average Return (Train)", np.mean(
        agent.stats.get_values("episode_returns")))
