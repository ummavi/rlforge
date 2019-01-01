import numpy as np
import tensorflow as tf

from abc import ABC
from abc import abstractmethod

if not tf.executing_eagerly():
    tf.enable_eager_execution()


class NetworkBlock(tf.keras.Model, ABC):
    """Defined an abstract base network class.
    """
    def __init__(self, params):
        tf.keras.Model.__init__(self)

        self.params = dict(self.default_params)
        self.params.update(params)

        self._layers = []
        self.build_block()

    @abstractmethod
    def build_block(self):
        """Builds the block of the network. 
        """
        raise NotImplementedError()


    def __call__(self, block_input):
        """Forward pass of the network block
        """
        prev_y = block_input
        for layer in self._layers:
            prev_y = layer(prev_y)
        return prev_y
    
    def clone(self):
        """Creates an identical network block with the same structure
        Useful when creating target networks.
        """
        return self.__class__(self.params)
        


class ConvBlock(NetworkBlock):
    default_params = dict(n_filters=[32, 64, 64],
                            filter_sizes=[8, 4, 3],
                            strides=[4, 2, 1],
                            activation=tf.nn.relu,
                            flatten_output=True)
        
    def __init__(self, params=None):
        """Builds a convolutional block. 
        Default network parameters for the CNN layer(s).
            returns the Nature DQN config
        """
        NetworkBlock.__init__(self, params)

    def build_block(self):
        """Builds the CNN block of the network. 
        """
        params_iter = zip(self.params["n_filters"],
                        self.params["filter_sizes"],
                        self.params["strides"])
        for fs,ks,strides in params_iter :
            layer = layers.Conv2D(fs, ks, strides=strides,
                                activation=self.params["activation"])
            self._layers.append(layer)

        if self.params["flatten_output"]:
            self._layers.append(tf.keras.layers.Flatten())

class DenseBlock(NetworkBlock):
    """Builds a block of dense layers
    """
    default_params = dict(layer_sizes=[256, 256], 
                          activation=tf.nn.relu,
                          weight_initializer="glorot_uniform")

    def __init__(self, params):
        NetworkBlock.__init__(self, params)

    def build_block(self):
        """Builds the conv layers followed by the dense
        """
        for layer_size in self.params["layer_sizes"]:
            layer = tf.keras.layers.Dense(layer_size, 
                                    activation=self.params["activation"],
                                    kernel_initializer=self.params["weight_initializer"])
            self._layers.append(layer)

class Sequential(tf.keras.Sequential):
    """Builds a sequential block composed of other blocks
    """
    def __init__(self, blocks):
        tf.keras.Sequential.__init__(self)

        self._blocks = blocks
        for block in self._blocks:
            self.add(block)

    def __call__(self, state_batch):
        """Forward pass of the entire sequential block
        """
        prev_y = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        for block in self._blocks:
            prev_y = block(prev_y)
        return prev_y

    def clone(self):
        """Clones the entire sequential block 
        """
        return Sequential([block.clone() for block in self._blocks])
