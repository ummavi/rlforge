import copy
import numpy as np
import tensorflow as tf

from rlforge.common.networks import DenseBlock, Sequential
from rlforge.common.networks import SplitOutputDenseBlock
from rlforge.common.utils import discounted_returns
from rlforge.common.utils import Episode


class ValueFunction:
    def __init__(self, model, optimizer, default_loss_fn, gamma):

        self.model = model
        self.gamma = gamma
        self.optimizer = optimizer
        self.loss_fn = default_loss_fn

    def set_loss_fn(self, loss_fn):
        """Set the loss function
        """
        self.loss_fn = loss_fn

    def __call__(self, states):
        """ Redirect call to the model
        """
        return self.model(states)

    def update_v(self, states, targets):
        """Perform one update step of the network
        """
        with tf.GradientTape() as tape:
            preds = self.v(states)
            losses = self.loss_fn(tf.squeeze(targets), tf.squeeze(preds))

            grads = tape.gradient(losses, self.model.trainable_weights)
            self.optimizer.apply_gradients(
                zip(grads, self.model.trainable_weights))
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
        all_returns = all_returns*np.float32(np.invert(all_dones))
        return (np.vstack(all_states), all_returns)

    def update_mc(self, episodes):
        """Perform a Monte-Carlo update of the network

        Parameters:
        episodes (list of Episode)
        """
        if self.optimizer is None:
            raise AssertionError("Did not pass `optimizer`  \
                                    when creating VNetworkDense")
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
        v_sns = np.ravel(self.v(states[self.n_steps:]))
        v_sns = np.hstack((v_sns[:-1], np.zeros(self.n_steps)))
        v_sns_discounted = self.gamma**self.n_steps * v_sns

        q_sts = trunc_returns + v_sns_discounted
        return q_sts

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

            v_sns = self.model.v(state_ns)
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
        self.model.reset()

    def q(self, states):
        raise Exception("Q not defined for " + str(self.__class__))

    def v(self, states):
        return self.model(states)

    def policy(self, states):
        raise Exception("Policy not defined for " + str(self.__class__))

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    @property
    def trainable_weights(self):
        return self.model.trainable_weights


class ValueFunctionDense(ValueFunction):
    def build_network(self, network_config, n_outputs):
        """Build a dense network according to the config. specified
        """
        # Copy the default network configuration (weight initialization)
        final_layer_config = dict(network_config)
        final_layer_config.update(
            dict(layer_sizes=[n_outputs], activation="linear"))
        return Sequential([
            DenseBlock(params=network_config),
            DenseBlock(params=final_layer_config)
        ])


class VNetworkDense(ValueFunctionDense):
    def __init__(self, network_config, gamma, optimizer=None, n_steps=1):
        """V-Network with only dense layers

        n_steps: MC parameter for N-Step returns
        """

        model = self.build_network(network_config, n_outputs=1)
        self.n_steps = n_steps
        default_loss_fn = tf.losses.mean_squared_error

        ValueFunction.__init__(self, model, optimizer, default_loss_fn, gamma)


class ValuePolicyNetworkDense(ValueFunction):
    def __init__(self,
                 network_config,
                 output_sizes,
                 gamma,
                 optimizer=None,
                 n_steps=1):
        """Merged Policy-V-Network with only dense layers

        n_steps: MC parameter for N-Step returns
        """

        model = self.build_network(network_config, output_sizes)
        self.n_steps = n_steps
        default_loss_fn = tf.losses.mean_squared_error

        ValueFunction.__init__(self, model, optimizer, default_loss_fn, gamma)

    def build_network(self, network_config, output_sizes):
        """Build a dense network according to the config. specified
        """
        # Copy the default network configuration (weight initialization)
        assert type(output_sizes) is list
        final_layer_config = dict(network_config)
        final_layer_config.update(
            dict(output_sizes=output_sizes, activation="linear"))

        return Sequential([
            DenseBlock(params=network_config),
            SplitOutputDenseBlock(params=final_layer_config)
        ])

    def q(self, states):
        raise Exception("V not defined for " + str(self.__class__))

    def v(self, states):
        return self.model(states)[-1]

    def policy(self, states):
        return self.model(states)[0]


class QNetworkDense(ValueFunctionDense):
    def __init__(self, n_actions, network_config, optimizer=None, gamma=0.98):
        """Q-Network with only dense layers
        """
        model = self.build_network(network_config, n_outputs=n_actions)
        self.n_actions = n_actions
        default_loss_fn = tf.losses.mean_squared_error
        ValueFunction.__init__(self, model, optimizer, default_loss_fn, gamma)

    def update_q(self, states, actions, targets):
        """Perform one update step of the network
        """
        with tf.GradientTape() as tape:
            preds = self.model.q(states)
            q_selected = tf.reduce_sum(
                preds * tf.one_hot(actions, self.n_actions), axis=-1)
            losses = self.loss_fn(tf.squeeze(targets), tf.squeeze(q_selected))
            grads = tape.gradient(losses, self.model.trainable_weights)
            self.optimizer.apply_gradients(
                zip(grads, self.model.trainable_weights))
        return preds, losses

    def policy(self, states):
        raise Exception("Policy not defined for " + str(self.__class__))

    def q(self, states):
        return self.model(states)

    def v(self, states):
        raise Exception("V not defined for " + str(self.__class__))
