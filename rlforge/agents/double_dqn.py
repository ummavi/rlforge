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
    
    See examples/double_dqn.py for example agent
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
