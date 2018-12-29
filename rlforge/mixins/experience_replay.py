import numpy as np 
from rlforge.mixins.base_mixin import BaseMixin

class ExperienceReplayMX(BaseMixin):
    """Experience Replay mixin 
    """
    def __init__(self, replay_buffer_size, minibatch_size):
        """
        """
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
    def __init__(self,max_size):
        self._max_size = max_size
        self._buffer = []
        self._next_idx = 0


    def append(self, state, action, reward, state_n,
               done, action_n=None):
        """Adds a transition to replay buffer
        """

        if action_n is not None:
            data = (state, action, reward, state_n, done, action_n)
        else:
            data = (state, action, reward, state_n, done)

        if len(self._buffer) < self._max_size:
            self._buffer.append(data)
        else:
            self._buffer[self._next_idx] = data

        self._next_idx = (self._next_idx + 1) % self._max_size

    def sample(self, batch_size):
        """Sample `batch_size` transitions from the experience replay
        """
        idxs = np.random.choice(len(self._buffer), batch_size)
        batch = [self._buffer[i] for i in idxs]
        return map(list, zip(*batch))

    def __len__(self):
        return len(self._buffer)