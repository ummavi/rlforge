import copy

from rlforge.common.networks import DenseBlock, Sequential


class PolicyNetwork:
    def __init__(self, model, gamma):
        self.model = model
        self.gamma = gamma

    def __call__(self, states):
        """ Redirect call to the model
        """
        return self.model(states)

    def clone(self):
        """Clone the whole value function wrapper
        """
        return copy.copy(self)

    def reset(self):
        self.model.reset()

    def q(self, states):
        raise Exception("Q not defined for "+str(self.__class__))

    def v(self, states):
        raise Exception("V not defined for "+str(self.__class__))

    def policy(self, states):
        return self.model(states)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    @property
    def trainable_weights(self):
        return self.model.trainable_weights


class PolicyNetworkDense(PolicyNetwork):
    def __init__(self, n_actions, network_config, gamma=0.98):

        model = self.build_network(network_config, n_outputs=n_actions)
        PolicyNetwork.__init__(self, model, gamma)

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
