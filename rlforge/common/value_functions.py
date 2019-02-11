import copy
import numpy as np
import tensorflow as tf

from rlforge.common.networks import DenseBlock, Sequential
from rlforge.common.utils import discounted_returns
from rlforge.common.utils import Episode


class ValueFunctionBase:
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
            preds = self.model(states)
            losses = self.loss_fn(tf.squeeze(targets), tf.squeeze(preds))

            grads = tape.gradient(losses, self.model.trainable_weights)
            self.optimizer.apply_gradients(
                zip(grads, self.model.trainable_weights))
        return preds, losses

    def clone(self):
        """Clone the whole value function wrapper
        """
        return copy.copy(self)

    def reset(self):
        self.model.reset()


class ValueFunctionDense(ValueFunctionBase):
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


class QNetworkDense(ValueFunctionDense):
    def __init__(self, n_actions, network_config, optimizer=None, gamma=0.98):
        """Simple Q-Network with only dense layers
        """
        model = self.build_network(network_config, n_outputs=n_actions)
        self.n_actions = n_actions
        default_loss_fn = tf.losses.mean_squared_error
        ValueFunctionBase.__init__(self, model, optimizer, default_loss_fn,
                                   gamma)

    def update_q(self, states, actions, targets):
        """Perform one update step of the network
        """
        with tf.GradientTape() as tape:
            preds = self.model(states)
            q_selected = tf.reduce_sum(
                preds * tf.one_hot(actions, self.n_actions), axis=-1)
            losses = self.loss_fn(tf.squeeze(targets), tf.squeeze(q_selected))
            grads = tape.gradient(losses, self.model.trainable_weights)
            self.optimizer.apply_gradients(
                zip(grads, self.model.trainable_weights))
        return preds, losses

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)


class VNetworkDense(ValueFunctionDense):
    def __init__(self, network_config, optimizer=None, gamma=0.98, n_steps=1):
        """Simple V-Network with only dense layers

        n_steps: MC parameter for N-Step returns
        """

        model = self.build_network(network_config, n_outputs=1)
        self.n_steps = n_steps
        default_loss_fn = tf.losses.mean_squared_error

        ValueFunctionBase.__init__(self, model, optimizer, default_loss_fn,
                                   gamma)

    def generate_mc_targets(self, episodes):
        """Process and generate MC targets of the network
        """
        all_states, all_returns = [], []

        for e in episodes:
            returns = discounted_returns(e.rewards, self.gamma)
            all_states += e.observations[:-1]
            all_returns += returns
        return (np.vstack(all_states), np.vstack(all_returns))

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

    def generate_td_targets(self, batch):
        """Process and generate TD targets
        """
        if type(batch[0]) is Episode:
            all_states, v_all_targets = [], []
            for e in batch:
                states, rewards = e.observations, e.rewards
                # Get V(s') where s' = s_{t+n}.
                # V(s_{T, T+1, T+2.....})=0 where s_T is the terminal state
                v_sns = np.ravel(self.model(states[self.n_steps:]))
                v_sns = np.hstack((v_sns[:-1], np.zeros(self.n_steps)))

                trunc_returns = discounted_returns(
                    rewards, self.gamma, n_steps=self.n_steps)

                discounted_v_sn = self.gamma**(self.n_steps + 1) * v_sns
                v_targets = trunc_returns + discounted_v_sn

                all_states += states[:-1]  # Don't update s_T
                v_all_targets += [v_targets]
            return (all_states, v_all_targets)
        else:
            states, _, rewards, state_ns, dones = zip(*batch)
            rewards, is_not_term = np.float32(rewards), np.float32(
                np.invert(dones))

            v_sns = self.model(state_ns)
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
