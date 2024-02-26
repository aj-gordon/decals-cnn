""""
Name: decals_net.py
Author: aj-gordon

CNN for the detection of tidal features around galaxies

"""

import sys

# import necessary layers from tensorflow
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input, Activation  # type: ignore
from tensorflow.keras.models import Model  # type: ignore


class DecalsNet(object):
    """
    The Convolutional Neural Network used in this work, Walmsley et al. (2019), and Dominguez Sanchez et al. (2023)
    
    Inputs:
        input_shape (tuple) - the shape of the images input into the CNN
        kernel_shape (tuple) - the shape of the convolutional kernels
        pool_shape (tuple) - the shape of the pooling layer kernels
        conv_activation (str) - the activation function to use in convolutional blocks
        dense_activation (str) - the activation function to use in dense blocks
        output_activation (str) - the activation function for the final output layer
        conv_nodes (tuple) - the number of nodes in each convolutional block (length of the tuple dictates number of blocks)
        dense_nodes (tuple) - the number of nodes in hidden dense blocks (length of the tuple dictates number of blocks)
        output_nodes (int) - the number of nodes in the final output layer
        use_dropout (bool) - whether or not to use dropout between dense blocks
        dropout_ratio (float) - if using dropout the probability of removing neurons
        name (str) - a name for the model
    """

    def __init__(
        self,
        input_shape: tuple,
        kernel_shape: tuple = (3, 3),
        pool_shape: tuple = (2, 2),
        conv_activation: str = "relu",
        dense_activation: str = "relu",
        output_activation: str = "sigmoid",
        conv_nodes: tuple = (32, 48, 64),
        dense_nodes: tuple = (64,),
        output_nodes: int = 4,
        use_dropout: bool = True,
        dropout_ratio: float = 0.5,
        name: str = "Decals-net",
    ):
        # input layer
        i = Input(shape=input_shape)
        x = i

        # add convolutional blocks
        for node in conv_nodes:
            x = self._conv_block(x, node, kernel_shape, pool_shape, conv_activation)

        # flatten the model
        x = Flatten()(x)

        # add the dense blocks for classification
        for node in dense_nodes:
            x = self._dense_block(x, node, dense_activation, use_dropout, dropout_ratio)

        # construct the output block
        x = Dense(output_nodes)(x)
        o = Activation(output_activation)(x)

        # convert to a tensorflow model
        self.model = Model(i, o, name=name)

    def _conv_block(
        self, x, nodes: int, kernel_shape: tuple, pool_shape: tuple, activation: str
    ):
        """
        A convolutional layer block: conv2D + activation + maxpooling2D - internal use only
        
        Inputs:
            nodes (int) - number of nodes in the convolutional layer
            kernel_shape (tuple) - the shape of the convolutional kernel
            pool_shape (tuple) - the shape of the pooling layer kernel
            activation (str) - the choice of activation function
        """

        # convolutional layer
        x = Conv2D(nodes, kernel_shape)(x)

        # activation layer
        x = Activation(activation)(x)

        # pooling layer
        x = MaxPooling2D(pool_shape)(x)

        return x

    def _dense_block(
        self, x, nodes: int, activation: str, use_dropout: bool, dropout: float = None
    ):
        """
        A dense layer block: dense layer + activation + dropout - internal use only
        
        Inputs:
            nodes (int) - number of nodes in the dense layer
            activation (str) - the choice of activation function
            use_dropout (bool) - whether or not to use drop out between this block and the next
            dropout (float) - the dropout rate if using it
        """

        # dense layer
        x = Dense(nodes)(x)

        # activation layer
        x = Activation(activation)(x)

        # dropout
        if use_dropout:
            x = Dropout(dropout)(x)

        return x


if __name__ == "__main__":
    sys.exit("Error: this code is not intended to be run as a main file.")
