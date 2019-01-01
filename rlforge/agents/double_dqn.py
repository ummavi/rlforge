import numpy as np
import tensorflow as tf

from rlforge.agents.dqn import DQNAgent

if not tf.executing_eagerly():
    tf.enable_eager_execution()


class DoubleDQNAgent(DQNAgent):
    """Double Deep-Q Learning Agent

    Implementation of the Double DQN algorithm 
    (https://arxiv.org/abs/1509.06461)


    DQN Targets:
    Y =  r + gamma * max_a(Q_target(s_new,a))

    DoubleDQN Targets:
    Y = r + gamma * Q_target(s_new, argmax_a(Q(s_new,a)))

    """

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

        best_sn_actions = np.argmax(self.model(state_ns), axis=-1)

        target_preds = self.target_model(state_ns)
        target_preds = tf.reduce_sum(
            target_preds * tf.one_hot(best_sn_actions, self.env.n_actions), axis=-1)

        q_target = rewards + (self.gamma * target_preds * is_not_terminal)

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
    from rlforge.common.value_functions import QNetworkDense
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
    agent = DoubleDQNAgent(env, q_network, policy_learning_rate=0.0001,
                           target_network_update_freq=300,
                           replay_buffer_size=10000,
                           gamma=0.8,
                           eps=0.2,
                           minibatch_size=128)
    train(agent, 250)
    print("Average Return (Train)", np.mean(
        agent.stats.get_values("episode_returns")))
