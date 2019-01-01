from rlforge.common.networks import DenseBlock, Sequential


def QNetworkDense(n_actions, dense_config):
    """Simple Q-Network with only dense layers
    """
    final_layer_config = dict(dense_config)
    final_layer_config.update(dict(layer_sizes=[n_actions],
                                   activation="linear"))
    return Sequential([DenseBlock(params=dense_config),
                       DenseBlock(params=final_layer_config)])
