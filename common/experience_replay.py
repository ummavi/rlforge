import numpy as np 


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