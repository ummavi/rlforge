import copy
import numpy as np

import chainer
from chainer import functions as F

from rlforge.common.networks import DenseBlock
from rlforge.common.networks import SplitOutputDenseBlock

from rlforge.common.utils import one_hot
from rlforge.common.utils import Episode
from rlforge.common.utils import discounted_returns


class QFunctionBase(chainer.Chain):
    def __init__(self, gamma,
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

    def update_q(self, states, actions, targets):
        """Perform one update step of the network
        """
        preds = self.q(states)
        q_selected = F.sum(
            preds * one_hot(actions, self.n_actions), axis=-1)
        losses = self.loss_fn(F.squeeze(targets), F.squeeze(q_selected))
        grads = tape.gradient(losses, self.model.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_weights))
        return preds, losses

    def policy(self, states):
        raise Exception("Policy not defined for " + str(self.__class__))

    def q(self, states):
        return self.forward(states)

    def v(self, states):
        raise Exception("V not defined for " + str(self.__class__))


class QNetworkDense(QFunctionBase):
    def __init__(self, n_actions, hidden_config, optimizer=None, gamma=0.98):
        """Q-Network with only dense layers
        """
        model = self.build_network(hidden_config, n_outputs=n_actions)
        self.n_actions = n_actions
        QFunctionBase.__init__(self, model, optimizer, gamma)

