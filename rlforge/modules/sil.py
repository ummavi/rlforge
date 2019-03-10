import chainer
import numpy as np

import chainer.functions as F

from rlforge.common.utils import Episode, discounted_returns

from rlforge.agents.a2c import A2CAgent
from rlforge.modules.base_mixin import BaseMixin
from rlforge.modules.experience_replay import ReplayBuffer


class SILExperienceReplayMX(BaseMixin):
    """Custom Experience replay module to calculate and store (s_t,a_t,R_t)
    R_t requires knowledge of r_{t+1}...r_{inf}. Therefore, add_to_replay
        is a post episode hook
    """

    def __init__(self, replay_buffer_size, minibatch_size):
        BaseMixin.__init__(self)
        self.minibatch_size = minibatch_size
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.post_episode_hooks.append(self.add_to_replay)

    def add_to_replay(self, global_episode_ts, episode_data):
        """
        post_step_hook Parameters:
            global_step_ts (int)
            step_data (Episode)
        """
        e = episode_data
        states, actions, rewards = e.observations, e.actions, e.rewards
        returns = discounted_returns(rewards, self.gamma)

        for s_t, a_t, R_t in zip(states, actions, returns):
            self.replay_buffer.append(s_t, a_t, R_t)

    def get_train_batch(self):
        """
        Overwrite the default get_train_batch to use a mini-batch from ER.
        """
        return self.replay_buffer.sample(self.minibatch_size)


class SILMX(SILExperienceReplayMX, BaseMixin):
    """
    Self-Imitation Learning paper implementation as described in
    (https://arxiv.org/abs/1806.05635)

    This is implemented as a module that can be inherited by any 
    Actor-Critic agent (whose model() returns policy output and value output)

    See examples/sil.py for example agent 

    Paper summary link will be added here shortly.
    """

    def __init__(self,
                 sil_buffer_size,
                 sil_minibatch_size,
                 sil_n_train_steps,
                 sil_value_coeff,
                 sil_run_every=1):

        BaseMixin.__init__(self)
        SILExperienceReplayMX.__init__(self, sil_buffer_size,
                                       sil_minibatch_size)

        self.sil_run_every = sil_run_every
        self.sil_value_coeff = sil_value_coeff
        self.sil_n_train_steps = sil_n_train_steps

        self.post_episode_hooks.append(self.learn_sil)
        self.reset_hooks.append(self.sil_reset)

    def learn_sil_step(self, batch):
        """Perform one off-policy SIL update

        Parameters:
        batch: sil_minibatch_size of (s,a,R) tuples.
        """
        states, actions, returns = batch
        numerical_prefs, v_sts = self.model(states)
        logprobs = self.logprobs(numerical_prefs, actions)
        returns = np.float32(returns)

        clipped_returns = F.maximum(returns - v_sts, np.zeros_like(returns))

        loss_sil_policy = -F.sum(logprobs * clipped_returns)
        loss_sil_value = 0.5 * F.sum(F.square(clipped_returns))

        losses = loss_sil_policy + self.sil_value_coeff * loss_sil_value

        self.model.cleargrads()
        losses.backward()
        self.optimizer.update()

    def learn_sil(self, global_episode_ts, episode_data):
        """Perform one round of SIL training composed of
        sil_n_train_step batches of off-policy training

        post_step_hook Parameters:
            global_step_ts (int)
            episode_data (Episode)
        """

        if global_episode_ts % self.sil_run_every != 0:
            return

        for m in range(self.sil_n_train_steps):
            minibatch = self.get_train_batch()
            self.learn_sil_step(minibatch)

    def sil_reset(self):
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
