import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe


class QNetwork(tf.keras.Model):
    """
    Default netwok parameters are the same as ATARI DQN.
    """
    cnn_params = dict(n_filters=[32, 64, 64], filter_sizes=[8, 4, 3],
                      strides=[4, 2, 1], activation=tf.nn.relu)
    dense_params = dict(layer_sizes=[256, 256], activation=tf.nn.relu)

    def __init__(self, n_actions, d_observations, dense_params, cnn_params=None):
        """
        Params:
        dense_params (dict): updates for the dense layer params
        cnn_params (dict or None): `None` disables the CNN. Dict
            updates the default (ATARI DQN) parameters
        """

        super(QNetwork, self).__init__()
        
        self.n_actions = n_actions
        self.d_observations = np.reshape(d_observations,-1)
        self.dense_params.update(dense_params)

        if cnn_params is None:
            self.cnn_params = None
        else:
            self.cnn_params.update(cnn_params)

        self.build_network()


    def build_network(self):
        """Builds the conv layers followed by the dense
        """
        self.cnn_layers, self.dense_layers = [], []
        if self.cnn_params is not None:
            self.cnn_layers = []
            c = self.cnn_params
            for i,(fs,ks,strides) in enumerate(zip(c["n_filters"],c["filter_sizes"],c["strides"])):
                layer = layers.Conv2D(fs, ks, strides=strides,
                                                activation=c["activation"])
                self.cnn_layers.append(layer)

            #Flatten output after the Conv layers.
            self.cnn_layers.append(tf.keras.layers.Flatten())

        c = self.dense_params
        for layer_size in c["layer_sizes"]:
            layer = tf.keras.layers.Dense(layer_size, activation=c["activation"])
            self.dense_layers.append(layer)

        layer = tf.keras.layers.Dense(self.n_actions)
        self.dense_layers.append(layer)

    def call(self, state_batch):
        """Forward pass of the Q-Function network

        Returns:
        Q
        """
        prev_y = np.reshape(state_batch,[-1]+list(self.d_observations))
        prev_y = tf.convert_to_tensor(prev_y, dtype=tf.float32)
        
        for layer in self.cnn_layers:
            prev_y = layer(prev_y)
        
        for layer in self.dense_layers:
            prev_y = layer(prev_y)
        output = prev_y
        return output
    
    def clone(self):
        """Creates an identical network with the same structure
        """
        return QNetwork(self.n_actions,self.d_observations,self.dense_params,self.cnn_params)
        