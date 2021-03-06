import numpy as np

import chainer
from chainer import functions as F
from functools import partial

from rlforge.agents.base_agent import BaseAgent
from rlforge.modules.policies import EpsilonGreedyPolicyMX
from rlforge.modules.experience_replay import ExperienceReplayMX
from rlforge.modules.target_network import TargetNetworkMX


class DQNAgent(EpsilonGreedyPolicyMX, ExperienceReplayMX, TargetNetworkMX,
               BaseAgent):
    """Deep-Q Network agent

    Implementation of the DQN agent as described in the nature paper
    (https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)


    * Huber Loss
    * Epsilon-greedy Policy (EpsilonGreedyPolicyMX)
    * Experience Replay (ExperienceReplayMX)
    * Uses Target Network to compute Q* for the target (TargetNetworkMX)

    Note: Python inheritance from R-L. List BaseAgent at the end

    See examples/dqn.py for example agent

    """

    def __init__(self,
                 environment,
                 q_function,
                 replay_buffer_size,
                 target_network_update_freq,
                 minibatch_size=32,
                 gamma=0.98,
                 ts_start_learning=200,
                 eps=0.2,
                 eps_schedule=None,
                 eps_start=None,
                 eps_end=None,
                 ts_eps_end=None,
                 experiment=None):

        self.network = q_function
        loss_fn = partial(F.huber_loss, delta=1.0)
        self.network.set_loss_fn(loss_fn)

        self.gamma = gamma
        self.ts_start_learning = ts_start_learning

        BaseAgent.__init__(self, environment, experiment)
        ExperienceReplayMX.__init__(self, replay_buffer_size, minibatch_size)
        TargetNetworkMX.__init__(self, target_network_update_freq)
        EpsilonGreedyPolicyMX.__init__(
            self,
            eps_fixed=eps,
            eps_schedule=eps_schedule,
            eps_start=eps_start,
            eps_end=eps_end,
            ts_eps_end=ts_eps_end)

        # DQN trains after every step so add it to the post_episode hook
        self.post_step_hooks.append(self.learn)

        self.model_list = [self.network]

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

        q_stars = F.max(self.target_network(state_ns), axis=-1)
        q_targets = rewards + (self.gamma * q_stars * is_not_terminal)

        preds, losses = self.network.update_q(states, actions, q_targets)

        self.logger.log_scalar("step_losses", float(losses.array))
        self.logger.log_scalar("step_mean_q", np.mean(preds.array))
