import numpy as np

import chainer
from chainer import functions as F
from chainer import links as L

from abc import ABC
from abc import abstractmethod


class NetworkBlock(chainer.Chain, ABC):
    """Defined an abstract base network class.
    """

    def __init__(self, params=None, build_block=False):
        """
        Parameters:
            params(optional): params passed to the block
            build_block: call build_block() on initialization
        """
        chainer.Chain.__init__(self)

        if type(params["activation"]) is str:
            params["activation"] = eval("F." + params["activation"])

        # Update default block parameters, if any
        self.b_params = dict(self.default_params)
        self.b_params.update(params)

        if build_block:
            self.build_block()

    @abstractmethod
    def build_block(self):
        """Builds the block of the network
        """
        raise NotImplementedError()

    def forward(self, x):
        """Forward pass of the network block
        """
        if type(x) is list:
            x = np.float32(x)
        prev_y = x
        for layer in self.layers:
            prev_y = self.b_params["activation"](layer(prev_y))
        return prev_y

    def reset(self):
        """Reset the weights (if any) of the block
        """
        raise NotImplementedError()
        self.layers = []
        self.build_block()


class DenseBlock(NetworkBlock):
    """Builds a block of dense layers
    """
    default_params = dict(
        layer_sizes=[256, 256],
        activation=F.tanh,
        weight_initializer=chainer.initializers.GlorotUniform(1.0))

    def __init__(self, params):
        NetworkBlock.__init__(self, params)

    def build_block(self):
        """Builds the FC/Dense block
        """
        with self.init_scope():
            layers = []
            for layer_size in self.b_params["layer_sizes"]:
                layer = L.Linear(
                    in_size=None,
                    out_size=layer_size,
                    initialW=self.b_params["weight_initializer"])
                layers.append(layer)
            self.layers = chainer.ChainList(*layers)


class SplitOutputDenseBlock(NetworkBlock):
    """Defines a split output dense block
    Useful wrapper around final layers when they output
    logically separable outputs.

    This allows a clean read interface like
    means, stds = model(inputs)
    """
    default_params = dict(
        split_axis=1,
        activation=None,
        output_sizes=[20, 20],
        weight_initializer=chainer.initializers.GlorotUniform(1.0))

    def __init__(self, params):
        NetworkBlock.__init__(self, params)

    def build_block(self):
        """Builds the conv layers followed by the dense
        """
        with self.init_scope():
            self.op_layer = L.Linear(
                in_size=None,
                out_size=sum(self.b_params["output_sizes"]),
                initialW=self.b_params["weight_initializer"])

    def forward(self, x):
        output_flat = self.op_layer(x)
        sections = self.b_params["output_sizes"][:-1]
        return F.split_axis(output_flat, indices_or_sections=sections, axis=1)