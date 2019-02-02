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

    def __call__(self, state_batch):
        """ Redirect call to the model
        """
        return self.model(state_batch)

    def update_model(self, state_batch, targets):
        """Perform one update step of the network
        """
        with tf.GradientTape() as tape:
            preds = self.model(state_batch)
            losses = self.loss_fn(tf.squeeze(targets), tf.squeeze(preds))

            grads = tape.gradient(losses, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads,
                                               self.model.trainable_weights))
        return preds, losses


class ValueFunctionDense(ValueFunctionBase):
    def build_network(self, network_config, n_outputs):
        """Build a dense network according to the config. specified
        """
        # Copy the default network configuration (weight initialization)
        final_layer_config = dict(network_config)
        final_layer_config.update(dict(layer_sizes=[n_outputs],
                                       activation="linear"))
        return Sequential([DenseBlock(params=network_config),
                           DenseBlock(params=final_layer_config)])


class QNetworkDense(ValueFunctionDense):

    def __init__(self, n_actions, network_config, optimizer, gamma=0.98):
        """Simple Q-Network with only dense layers
        """

        model = self.build_network(network_config, n_outputs=n_actions)
        default_loss_fn = tf.losses.mean_squared_error

        ValueFunctionBase.__init__(self, model, optimizer,
                                   default_loss_fn, gamma)


class VNetworkDense(ValueFunctionDense):
    def __init__(self, network_config, optimizer, gamma=0.98, n_steps=1):
        """Simple V-Network with only dense layers

        n_steps: MC parameter for N-Step returns
        """

        model = self.build_network(network_config, n_outputs=1)
        self.n_steps = n_steps
        default_loss_fn = tf.losses.mean_squared_error

        ValueFunctionBase.__init__(self, model, optimizer,
                                   default_loss_fn, gamma)

    def update_mc(self, episodes):
        """Perform a Monte-Carlo update of the network

        Parameters:
        episodes (list of Episode)
        """
        all_states, all_returns = [], []

        for e in episodes:
            returns = discounted_returns(e.rewards, self.gamma)
            all_states += e.observations[:-1]
            all_returns += returns

        self.update_model(np.vstack(all_states), np.vstack(all_returns))

    def update_td(self, batch):
        """Perform a td update of the network

        Parameters:
        batch (list of Episodes OR list of Transitions)

        TODO: n_steps=None 
        """

        if type(batch[0]) is Episode:
            all_states, v_all_targets = [], []
            for e in batch:
                states, rewards = e.observations, e.rewards
                # Get V(s') where s' = s_{t+n}.
                # V(s_{T, T+1, T+2.....})=0 where s_T is the terminal state
                v_sns = np.ravel(self.model(states[self.n_steps:]))
                v_sns = np.hstack((v_sns[:-1], np.zeros(self.n_steps)))

                trunc_returns = discounted_returns(rewards, self.gamma,
                                                   n_steps=self.n_steps)

                discounted_v_sn = self.gamma**(self.n_steps + 1) * v_sns
                v_targets = trunc_returns + discounted_v_sn

                all_states += states[:-1]  # Don't update s_T
                v_all_targets += [v_targets]

            self.update_model(all_states, v_all_targets)
        else:
            states, _, rewards, state_ns, dones = zip(*batch)
            rewards, is_not_term = np.float32(
                rewards), np.float32(np.invert(dones))

            v_sns = self.model(state_ns)
            v_target = rewards + (self.gamma * v_sns * is_not_term)

            self.update_model(states, v_target)