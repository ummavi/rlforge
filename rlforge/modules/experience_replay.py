import numpy as np
from rlforge.modules.base_mixin import BaseMixin


class ExperienceReplayMX(BaseMixin):
    """Experience Replay mixin 
    """

    def __init__(self, replay_buffer_size, minibatch_size):
        BaseMixin.__init__(self)
        self.minibatch_size = minibatch_size
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.post_step_hooks.append(self.add_to_replay)

    def add_to_replay(self, global_step_ts, step_data):
        """
        post_step_hook Parameters:
            global_step_ts (int)
            step_data (tuple): (s,a,r,s',done)
        """
        s, a, r, s_n, d = step_data
        self.replay_buffer.append(s, a, r, s_n, d)

    def get_train_batch(self):
        """
        Overwrite the default get_train_batch to use a mini-batch from ER.
        """
        return self.replay_buffer.sample(self.minibatch_size)


class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        self.next_loc = 0

    def append(self, state, action, reward, state_n, done, action_n=None):
        """Adds a transition to replay buffer
        """

        if action_n is not None:
            data = (state, action, reward, state_n, done, action_n)
        else:
            data = (state, action, reward, state_n, done)

        if len(self.buffer) < self.max_size:
            self.buffer.append(data)
        else:
            self.buffer[self.next_loc] = data

        self.next_loc = (self.next_loc + 1) % self.max_size

    def sample(self, batch_size):
        """Sample `batch_size` transitions from the experience replay
        """
        idxs = np.random.choice(len(self.buffer), batch_size)
        batch = [self.buffer[i] for i in idxs]
        return map(list, zip(*batch))

    def __len__(self):
        return len(self.buffer)
