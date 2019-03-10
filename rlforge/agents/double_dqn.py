import numpy as np

import chainer.functions as F
from rlforge.agents.dqn import DQNAgent
from rlforge.common.utils import one_hot

class DoubleDQNAgent(DQNAgent):
    """Double Deep-Q Learning Agent

    Implementation of the Double DQN algorithm
    (https://arxiv.org/abs/1509.06461)


    DQN Targets:
    Y =  r + gamma * max_a(Q_targets(s_new,a))

    DoubleDQN Targets:
    Y = r + gamma * Q_targets(s_new, argmax_a(Q(s_new,a)))

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
        rewards, is_not_terminal = np.float32(rewards), np.float32(
            np.invert(dones))

        best_sn_actions = np.argmax(self.network(state_ns), axis=-1)

        target_preds = self.target_network(state_ns)
        target_preds = F.sum(
            target_preds * one_hot(best_sn_actions, self.env.n_actions),
            axis=-1)

        q_targets = rewards + (self.gamma * target_preds * is_not_terminal)
        preds, losses = self.network.update_q(states, actions, q_targets)

        self.logger.log_scalar("step_losses", float(losses.array))
        self.logger.log_scalar("step_mean_q", np.mean(preds))
