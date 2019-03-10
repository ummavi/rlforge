import copy
import numpy as np

import chainer
from chainer import functions as F

from rlforge.networks.blocks import DenseBlock
from rlforge.networks.blocks import SplitOutputDenseBlock

from rlforge.common.utils import one_hot
from rlforge.common.utils import Episode
from rlforge.common.utils import discounted_returns


class QFunctionBase(chainer.Chain):
    def __init__(self, n_actions, gamma,
                 optimizer=None,
                 default_loss_fn=F.mean_squared_error):
        super().__init__()

        self.n_actions = n_actions
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
        losses = self.loss_fn(F.expand_dims(targets, 0),
                              F.expand_dims(q_selected, 0))

        self.cleargrads()
        losses.backward()
        self.optimizer.update()

        return preds, losses

    def policy(self, states):
        raise Exception("Policy not defined for " + str(self.__class__))

    def q(self, states):
        return self.forward(states)

    def v(self, states):
        raise Exception("V not defined for " + str(self.__class__))

    def clone(self):
        return self.copy("copy")


class QNetworkDense(QFunctionBase):
    def __init__(self, hidden_config, n_actions, gamma, optimizer=None):
        """Q-Network with only dense layers
        """
        QFunctionBase.__init__(self, n_actions, gamma, optimizer)

        self.build_network(hidden_config, n_outputs=n_actions)
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
