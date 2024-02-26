"""
Name: preprocessor.py
Author: aj-gordon
"""

# sys import
import sys

# generic imports
import numpy as np

# tensorflow
from tensorflow.keras import Model  # type: ignore
from tensorflow.keras.layers import (
    RandomFlip,
    RandomRotation,
    RandomTranslation,
    RandomZoom,
    Resizing,
)  # type: ignore


class Preprocessor(Model):
    """Preprocessor is a Tensorflow Model subclass"""

    def __init__(
        self,
        output_size: tuple,
        angle_mode: str = "full_range",
        flip_mode: str = "horizontal_and_vertical",
        angle: float = np.pi / 2,
        translation: float = 0.05,
        zoom: float = 0.1,
        fill_mode: str = "constant",
    ):
        """
        Data preprocessor and augmentation model - randomly transforms the data

        Inputs:
        output_size (int) - the shape to resisze the data to, height and width will be changed to \
            this shape, colour channels will remain unchanged
        flip_mode [opt] (str) - mode to randomly flip the images, choose 'horizonal_and_vertical', \
            'horizontal', or 'vertical'
        angle [opt] (float) - maximum angle in radians to randomly rotate the images
        translation [opt] (float) - percentage factor to translate the images
        zoom [opt] (float) - percentage factor to zoom the images
        fill_mode [opt] (str) - how to fill empty space

        Outputs:
        The object is a model - to call use Name() or Name.call()
        """
        # initialise
        super().__init__()

        self.angle_mode = angle_mode

        # define rotation factor in the appropriate form for the layer
        rotation_factor = angle / (2 * np.pi)

        # define the layers
        self.flip = RandomFlip(mode=flip_mode)

        if self.angle_mode == "full_range":
            self.rotate = RandomRotation(factor=rotation_factor, fill_mode=fill_mode)
        elif self.angle_mode == "90_degrees":
            self.rotate = [RandomRotation(factor=(-0.25, -0.25), fill_mode=fill_mode),
                            RandomRotation(factor=(0., 0.), fill_mode=fill_mode),
                            RandomRotation(factor=(+0.25, +0.25), fill_mode=fill_mode)]
        self.translate = RandomTranslation(
            height_factor=translation, width_factor=translation, fill_mode=fill_mode
        )
        self.zoom = RandomZoom(
            height_factor=zoom, width_factor=None, fill_mode=fill_mode
        )
        self.resize = Resizing(output_size[0], output_size[1])

    def call(self, x, augment: bool = True):
        """
        Call function for calling the model

        Inputs:
            x - data
            [optional] augment (bool) - choose whether or not to augment the data
        """

        # if augmenting then perform augmentations
        if augment:
            x = self.flip(x)
            if self.angle_mode == "full_range":
                x = self.rotate(x)
            elif self.angle_mode == "90_degrees":
                x = np.random.choice(self.rotate)(x)
            x = self.translate(x)
            x = self.zoom(x)

        # always resize the model to the desired shape
        x = self.resize(x)

        return x


if __name__ == "__main__":
    sys.exit("Error: this code is not intended to be run as a main file.")
