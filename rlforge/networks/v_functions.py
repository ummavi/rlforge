import copy
import numpy as np

import chainer
from chainer import functions as F

from rlforge.networks.blocks import DenseBlock
from rlforge.networks.blocks import SplitOutputDenseBlock

from rlforge.common.utils import Episode
from rlforge.common.utils import discounted_returns


class ValueFunctionBase(chainer.Chain):
    def __init__(self,
                 gamma,
                 optimizer=None,
                 default_loss_fn=F.mean_squared_error):
        super().__init__()

        self.gamma = gamma
        self.optimizer = optimizer
        self.loss_fn = default_loss_fn

    def set_loss_fn(self, loss_fn):
        """Set the loss function
        """
        self.loss_fn = loss_fn

    def update_v(self, states, targets):
        """Perform one update step of the network
        """
        if self.optimizer is None:
            raise AssertionError("Did not pass `optimizer` \
                                  when creating network? ")
        preds = self.v(states)
        losses = self.loss_fn(F.squeeze(targets), F.squeeze(preds))

        self.cleargrads()
        losses.backward()
        self.optimizer.update()

        return preds, losses

    def generate_mc_targets(self, episodes):
        """Process and generate MC targets of the network
        """
        all_states, all_returns, all_dones = [], [], []

        for e in episodes:
            returns = discounted_returns(e.rewards, self.gamma)
            all_states += e.observations[:-1]
            all_returns += returns
            all_dones += e.dones

        all_returns = np.vstack(all_returns)
        all_dones = np.vstack(all_dones)
        # Returns for terminal states are 0.
        all_returns = all_returns * np.float32(np.invert(all_dones))
        return (np.vstack(all_states), np.float32(all_returns))

    def update_mc(self, episodes):
        """Perform a Monte-Carlo update of the network

        Parameters:
        episodes (list of Episode)
        """
        states, targets = self.generate_mc_targets(episodes)
        self.update_v(states, targets)

    def nstep_bootstrapped_value(self, episode, n_steps=1):
        """Generate n_step bootstrapped value estimate

        Q(s_t,a_t) = r_t + γ*V(s_t+1) for n_steps=1
        Q(s_t,a_t) = r_t + γ*r_{t+2} + γ^2*V(s_{t+3}) for n_steps=2
        ...
        Q(s_t,a_t) = Σ_{i=t...t+n} r_i +γ^{n}*V(s_{t+n+1}) for n_steps=n

        TODO: Handle error when there are < n_steps states
        """
        states, rewards = episode.observations, episode.rewards

        trunc_returns = discounted_returns(
            rewards, self.gamma, n_steps=self.n_steps)

        # Get V(s') where s' = s_{t+n}.
        # V(s_{T, T+1, T+2.....})=0 where s_T is the terminal state
        v_sns = np.ravel(self.v(states[self.n_steps:]).array)
        v_sns = np.hstack((v_sns[:-1], np.zeros(self.n_steps)))
        v_sns_discounted = self.gamma**self.n_steps * v_sns

        q_sts = trunc_returns + v_sns_discounted
        return np.float32(q_sts)

    def generate_td_targets(self, batch):
        """Process and generate TD targets
        """
        if type(batch[0]) is Episode:
            all_states, v_all_targets = [], []
            for e in batch:
                v_targets = self.nstep_bootstrapped_value(e)

                all_states += e.observations
                # Value function for the terminal state is set to 0.0
                v_all_targets += [np.hstack((v_targets, 0.0))]

            return (np.float32(all_states), np.float32(v_all_targets))
        else:
            states, _, rewards, state_ns, dones = zip(*batch)
            rewards, is_not_term = np.float32(rewards), np.float32(
                np.invert(dones))

            v_sns = self.v(state_ns)
            v_target = rewards + (self.gamma * v_sns * is_not_term)
            return (states, v_target)

    def update_td(self, batch):
        """Perform a td update of the network

        Parameters:
        batch (list of Episodes OR list of Transitions)

        TODO: n_steps=None
        """
        if self.optimizer is None:
            raise AssertionError("Did not pass `optimizer`  \
                                    when creating VNetworkDense")
        states, targets = self.generate_td_targets(batch)
        self.update_v(states, targets)

    def clone(self):
        """Clone the whole value function wrapper
        """
        return copy.copy(self)

    def reset(self):
        raise NotImplementedError()
        self.reset()

    def q(self, states):
        raise Exception("Q not defined for " + str(self.__class__))

    def v(self, states):
        return self.forward(states)

    def policy(self, states):
        raise Exception("Policy not defined for " + str(self.__class__))

    def clone(self):
        return self.copy("copy")

class ValueNetworkDense(ValueFunctionBase):
    def __init__(self, hidden_config, gamma, optimizer=None, n_steps=1):
        """V-Network with only dense layers

        n_steps: MC parameter for N-Step returns
        """
        ValueFunctionBase.__init__(self, gamma, optimizer)

        self.n_steps = n_steps
        self.build_network(hidden_config, n_outputs=1)
        if optimizer is not None:
            self.optimizer.setup(self)

    def build_network(self, hidden_config, n_outputs):
        """Build a dense network according to the config. specified
        """
        with self.init_scope():
            # Copy the default network configuration (weight initialization)
            final_config = dict(hidden_config)
            final_config.update(
                dict(layer_sizes=[n_outputs], activation="identity"))

            blocks = [
                DenseBlock(params=hidden_config),
                DenseBlock(params=final_config)
            ]
            self.blocks = chainer.ChainList(*blocks)
            for block in self.blocks:
                block.build_block()

    def forward(self, x):
        x = np.float32(x)
        for block in self.blocks:
            x = block(x)
        return x


"""
-------------------
Combined Policy+Value functions for Actor-Critic functions defined below.
-------------------
"""


class ValuePolicyNetworkDense(ValueFunctionBase):
    def __init__(self,
                 hidden_config,
                 n_actions,
                 gamma,
                 optimizer=None,
                 n_steps=1):
        """Policy+Value function without shared parameters
        """
        assert type(n_actions) is int
        assert type(hidden_config) is dict
        ValueFunctionBase.__init__(self, gamma, optimizer)

        self.n_steps = n_steps
        self.build_network(hidden_config, n_actions)
        if optimizer is not None:
            self.optimizer.setup(self)

    def build_network(self, hidden_config, n_actions):
        """Build a dense network according to the config. specified
        """
        with self.init_scope():
            final_config = dict(hidden_config)
            final_config.update(
                dict(layer_sizes=[n_actions], activation="identity"))
            actor_blocks = [
                DenseBlock(params=hidden_config),
                DenseBlock(params=final_config)
            ]
            self.actor_blocks = chainer.ChainList(*actor_blocks)

            final_config = dict(hidden_config)
            final_config.update(dict(layer_sizes=[1], activation="identity"))
            critic_blocks = [
                DenseBlock(params=hidden_config),
                DenseBlock(params=final_config)
            ]
            self.critic_blocks = chainer.ChainList(*critic_blocks)

            for block in self.actor_blocks:
                block.build_block()

            for block in self.critic_blocks:
                block.build_block()

    def forward_partial(self, x, network="both"):
        x = np.float32(x)
        blocks = self.actor_blocks if network is "actor" else self.critic_blocks
        for block in blocks:
            x = block(x)
        return x

    def forward(self, x):
        return (self.policy(x), self.v(x))

    def q(self, states):
        raise Exception("V not defined for " + str(self.__class__))

    def v(self, states):
        v = self.forward_partial(states, network="critic")
        return v

    def policy(self, states):
        p = self.forward_partial(states, network="actor")
        return p


class ValuePolicyNetworkDenseShared(ValueNetworkDense):
    """Value function with shared parameters

    TODO: Cache self.forward() call to reuse for v and q
    """

    def __init__(self,
                 hidden_config,
                 n_actions,
                 gamma,
                 optimizer=None,
                 n_steps=1):
        """Shared Policy+Value function.
        """
        assert type(n_actions) is int
        assert type(hidden_config) is dict
        ValueFunctionBase.__init__(self, gamma, optimizer)
        self.n_steps = n_steps
        self.build_network(hidden_config, n_actions)
        if optimizer is not None:
            self.optimizer.setup(self)

    def build_network(self, hidden_config, n_actions):
        """Build a dense network according to the config. specified
        """
        with self.init_scope():
            # Copy the default network configuration (weight initialization)
            final_config = dict(hidden_config)
            final_config.update(
                dict(output_sizes=[n_actions, 1], activation="identity"))

            blocks = [
                DenseBlock(params=hidden_config),
                SplitOutputDenseBlock(params=final_config)
            ]
            self.blocks = chainer.ChainList(*blocks)
            for block in self.blocks:
                block.build_block()

    def q(self, states):
        raise Exception("V not defined for " + str(self.__class__))

    def v(self, states):
        return self.forward(states)[-1]

    def policy(self, states):
        return self.forward(states)[0]
