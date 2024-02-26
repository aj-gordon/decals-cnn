"""
Name: data.py
Author: aj-gordon

Definition of a data class to hold and handle image and table data for loading into the network during training, testing, or discovery

"""

# pylint: disable=import-error, wrong-import-position

# sys import
import sys

# package imports
import numpy as np
import pandas as pd

# tensorflow, dataset, and autotuning
import tensorflow as tf # type: ignore
from tensorflow.data import Dataset, AUTOTUNE  # type: ignore


# object to hold data
class Data(object):
    """
    Data object for holding and handling all of the necessary data

    Inputs:
        separate_paths (bool) - if false the table and image data are not contained in the same directory
        tablepath (str) - path to the table to read data from (after generalpath)
        impath (str) - path to the image files to read data (after generalpath if separate_paths is false)
                        alternatively the path to the image files not including the generalpath
        generalpath (str) - path to the overall directory containing the table and image data
        preprocessor (tf.keras.Model) - the image preprocessing model
        set_kind (str) - whether the set includes training, testing, validation, or unseen data. Select one of:
                            all - training, testing, and validation data
                            testing - only testing data and their labels
                            discover - only data is stored no labels
    """

    def __init__(
        self,
        tablepath: str,
        preprocessor: tf.keras.Model,
        imagepath: str = "path/to/images/",
        generalpath: str = "path/to/directory/",
        set_kind: str = "all",
        separate_paths: bool = False,
    ):


        # define a variable for the storage locations
        if separate_paths is False:
            self.imagepath = generalpath + imagepath
        elif separate_paths is True:
            self.imagepath = imagepath
        else:
            raise ValueError("SEPARATE_PATHS must be either True or False")

        self.tablepath = generalpath + tablepath
        self.set_kind = set_kind

        # set preprocessor model
        self.preprocessor = preprocessor

        # make the training, validation, and testing sets based on set_kind
        if self.set_kind == "all":
            # define the training data variable
            self.train_data = []

            # define the validation data variable
            self.val_data = []

            # define the testing data variable
            self.test_data = []
            self.test_data_names = []
            self.test_data_labels = []

            # define the training weight given to each class
            self.class_weights = {}

        elif self.set_kind == "testing":
            # define the testing data variable
            self.test_data = []
            self.test_data_names = []
            self.test_data_labels = []

        elif self.set_kind == "discover":
            # define the testing data variable
            self.test_data = []
            self.test_data_names = []

        else:
            # quit if wrong type
            sys.exit("Error: set_kind must be either 'all' or 'discover'.")

    ### vectorised functions ###

    def _add_filepath(self, image_name: str) -> str:
        """
        Function to add the filepath to the image name - internal use only
        
        adds self.impagepath to the image name from the table
        """

        # adds the filepath to the image name
        return self.imagepath + image_name

    def _load_image(self, fullimagepath: str):
        """
        Function to load in the images - internal use only
        
        Uses tensorflow to read in the PNG images and convert them to the proper float type for use
        """

        # read the images
        image = tf.io.read_file(fullimagepath)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        # return the image
        return image

    def _to_categorical(self, label, mode: str = "multi-label"):
        """
        Function to convert a unique label to one-hot encoding - internal use only
        
        converts labels from the visual inspection process to one-hot encoding
        """

        # transform a unique label into a one-hot encoded one
        categorical = np.array([0, 0, 0, 0])

        # if has a tidal arm structure
        if label in ["TA", "TF", "TS", "TD", "AF", "DA", "DS"]:
            categorical[0] = 1

        # if has a faint stream structure
        if label in ["FS", "TF", "FA", "FD", "AF", "DA", "DF"]:
            categorical[1] = 1

        # if has a shell symmetric structure
        if label in ["AS", "TS", "FA", "SD", "AF", "DS", "DF"]:
            categorical[2] = 1

        # if has an asymmetric diffuse structure
        if label in ["AD", "TD", "FD", "SD", "DA", "DS", "DF"]:
            categorical[3] = 1

        if mode == "multi-label":
            return categorical
        elif mode == "binary":
            if np.sum(categorical) > 0:
                return 1
            else:
                return 0
        else:
            sys.exit("Error: mode must be either 'multi-label' or 'binary'.")

    ### data loading method ###

    def load_data(
        self,
        batch_size: int,
        augment: bool = True,
        validation_split: float = 0.1,
        test_split: float = 0.2,
        filename_header: str = "filename",
        feature_header: str = "feature",
        mode: str = "multi-label",
    ):
        """
        Loads the data from png images to tensor

        Inputs:
        batch_size (int) - the size of each batch for the tensor
        augment (bool) - if true the training data will be augmented
        validation_split (float) - the ratio of training data (e.g. total - test) that will be saved for validation
        test_split (float) - the ratio of the whole dataset that will be saved for testing
        filename_header (str) - the column name in the table for the filenames
        feature_header (str) - the column name in the table for the features
        """

        # vectorize the functions
        _add_filepath = np.vectorize(self._add_filepath)
        _to_categorical = np.vectorize(self._to_categorical, signature="()->(n)")

        # if generating all data types
        if self.set_kind == "all":
            # extract files and labels from the table
            table_data = np.array(
                pd.read_csv(self.tablepath)[[filename_header, feature_header]]
            )

            # get full filepath of images
            files = _add_filepath(table_data[:, 0])

            # get one hot encoded or binary labels
            if mode == "binary":
                labels = np.where(
                    np.sum(_to_categorical(table_data[:, 1]), axis=1) >= 1, 1, 0
                )
            elif mode == "multi-label":
                labels = _to_categorical(table_data[:, 1])
            else:
                sys.exit("Error: mode must be either 'multi-label' or 'binary'.")

            # randomly select some testing images
            test_ids = np.random.choice(
                np.arange(len(files)), int(test_split * len(files)), replace=False
            )

            # lists of the test set files and labels
            test_files = files[test_ids]
            test_labels = labels[test_ids]
            self.test_data_labels = test_labels
            self.test_data_names = test_files

            # remove files and labels from the set
            new_files = np.delete(files, test_ids, axis=0)
            new_labels = np.delete(labels, test_ids, axis=0)

            # randomly select some validation images
            val_ids = np.random.choice(
                np.arange(len(new_files)),
                int(validation_split * len(new_files)),
                replace=False,
            )

            # lists of validation set files and labels
            val_files = new_files[val_ids]
            val_labels = new_labels[val_ids]

            # remove validation data to get training data
            train_files = np.delete(new_files, val_ids, axis=0)
            train_labels = np.delete(new_labels, val_ids, axis=0)

            # obtain the class weights
            class_totals = np.sum(train_labels, axis=0)
            num_samples = train_labels.shape[0]

            if mode == "binary":
                self.class_weights[0] = 0.5*num_samples/(num_samples - class_totals)
                self.class_weights[1] = 0.5*num_samples/class_totals
            elif mode == "multi-label":
                for i, total in enumerate(class_totals):
                    self.class_weights[i] = num_samples/(4*total)
            else:
                raise ValueError("MODE must be either 'binary' or 'multi-label'")

            # set up training data pipeline
            self.train_data = Dataset.from_tensor_slices((train_files, train_labels))
            self.train_data = self.train_data.shuffle(len(train_files))
            self.train_data = self.train_data.map(
                lambda x, y: (self._load_image(x), y), num_parallel_calls=AUTOTUNE
            )
            self.train_data = self.train_data.cache().repeat()
            self.train_data = self.train_data.batch(batch_size)

            # if augmenting create the model and augment the data
            self.train_data = self.train_data.map(
                lambda x, y: (self.preprocessor(x, augment=augment), y),
                num_parallel_calls=AUTOTUNE,
            )

            # prefetch
            self.train_data = self.train_data.prefetch(AUTOTUNE)

            # set up validation data pipeline in similar manner
            self.val_data = Dataset.from_tensor_slices((val_files, val_labels))
            self.val_data = self.val_data.map(
                lambda x, y: (self._load_image(x), y), num_parallel_calls=AUTOTUNE
            )
            self.val_data = self.val_data.cache()
            self.val_data = self.val_data.batch(batch_size)
            self.val_data = self.val_data.map(
                lambda x, y: (self.preprocessor(x, augment=False), y),
                num_parallel_calls=AUTOTUNE,
            )
            self.val_data = self.val_data.prefetch(AUTOTUNE)

            # set up testing data pipeline in similar manner
            self.test_data = Dataset.from_tensor_slices(test_files)
            self.test_data = self.test_data.map(
                self._load_image, num_parallel_calls=AUTOTUNE
            )
            self.test_data = self.test_data.cache()
            self.test_data = self.test_data.batch(batch_size)
            self.test_data = self.test_data.map(
                lambda x: (self.preprocessor(x, augment=False)),
                num_parallel_calls=AUTOTUNE,
            )
            self.test_data = self.test_data.prefetch(AUTOTUNE)

        elif self.set_kind == "testing":
            # extract files and labels from the table
            table_data = np.array(
                pd.read_csv(self.tablepath)[[filename_header, feature_header]]
            )

            self.test_data_labels = _to_categorical(table_data[:, 1])

            # get full filepath of images
            files = _add_filepath(table_data[:, 0])
            self.test_data_names = files

            # set up testing data pipeline in similar manner
            self.test_data = Dataset.from_tensor_slices(files)
            self.test_data = self.test_data.map(
                self._load_image, num_parallel_calls=AUTOTUNE
            )
            self.test_data = self.test_data.cache()
            self.test_data = self.test_data.batch(batch_size)
            self.test_data = self.test_data.map(
                lambda x: self.preprocessor(x, augment=False),
                num_parallel_calls=AUTOTUNE,
            )
            self.test_data = self.test_data.prefetch(AUTOTUNE)

        elif self.set_kind == "discover":
            # extract files and labels from the table
            table_data = np.array(pd.read_csv(self.tablepath)[filename_header])

            # get full filepath of images
            files = _add_filepath(table_data)
            self.test_data_names = files

            # set up testing data pipeline in similar manner
            self.test_data = Dataset.from_tensor_slices(files)
            self.test_data = self.test_data.map(
                self._load_image, num_parallel_calls=AUTOTUNE
            )
            self.test_data = self.test_data.cache()
            self.test_data = self.test_data.batch(batch_size)
            self.test_data = self.test_data.map(
                lambda x: self.preprocessor(x, augment=False),
                num_parallel_calls=AUTOTUNE,
            )
            self.test_data = self.test_data.prefetch(AUTOTUNE)

        else:
            # quit if wrong type
            sys.exit(
                "Error: encountered an unexpected error with set_kind - please select either\
                     'all', 'testing' or 'discover'."
            )


if __name__ == "__main__":
    sys.exit("Error: this code is not intended to be run as a main file.")
