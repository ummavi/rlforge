import copy
import chainer
import numpy as np

from rlforge.common.networks import DenseBlock


class PolicyNetwork:
    def __init__(self, gamma):
        self.gamma = gamma

    def clone(self):
        """Clone the whole value function wrapper
        """
        return copy.copy(self)

    def reset(self):
        self.reset()

    def q(self, states):
        raise Exception("Q not defined for "+str(self.__class__))

    def v(self, states):
        raise Exception("V not defined for "+str(self.__class__))

    def policy(self, states):
        return self.forward(states)


class PolicyNetworkDense(chainer.Chain, PolicyNetwork):
    def __init__(self, n_actions, hidden_config, gamma=0.98):
        PolicyNetwork.__init__(self, gamma)
        chainer.Chain.__init__(self)
        self.build_network(hidden_config, n_outputs=n_actions)

    def build_network(self, hidden_config, n_outputs):
        """Build a dense network according to the config. specified
        """
        with self.init_scope():
            # Copy the default network configuration (weight initialization)
            final_config = dict(hidden_config)
            final_config.update(dict(layer_sizes=[n_outputs], activation="tanh"))

            blocks = [DenseBlock(params=hidden_config),
                      DenseBlock(params=final_config)]
            self.blocks = chainer.ChainList(*blocks)
            for block in self.blocks:
                block.build_block()

    def forward(self, x):
        x = np.float32(x)
        for block in self.blocks:
            x = block(x)
        return x
