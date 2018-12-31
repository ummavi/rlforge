import numpy as np
import tensorflow as tf

from rlforge.common.networks import DenseBlock,Sequential

def QNetworkDense(n_actions,dense_config):
    """Simple Q-Network with only dense layers
    """
    return Sequential([DenseBlock(params=dense_config), 
                DenseBlock(params=dict(layer_sizes=[n_actions],
                                        activation="linear"))])

