"""
Name: read_args.py
Author: aj-gordon
"""

# import argparse
import argparse


def _recognise_boolean(inpt: bool or str) -> bool:
    """Function that recognises boolean values from the command line"""

    if isinstance(inpt, bool):
        return inpt
    elif inpt.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif inpt.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def read_train_arguments():
    """Function that reads the arguments for training from the command line"""

    parser = argparse.ArgumentParser(description="CNN training and testing")
    parser.add_argument(
        "table", type=str, help="The name of the table that contains the data"
    )
    parser.add_argument(
        "-s",
        "--TRAIN-SIZE",
        type=int,
        default=1050,
        help="The number of images per epoch",
    )
    parser.add_argument(
        "-e", "--EPOCHS", type=int, default=100, help="The number of training epochs"
    )
    parser.add_argument(
        "-b",
        "--BATCH-SIZE",
        type=int,
        default=75,
        help="The size of training image batches",
    )
    parser.add_argument(
        "-v",
        "--VERBOSE",
        choices=[0, 1, 2],
        type=int,
        default=2,
        help="0 - no updates, 1 - progress bar, 2 - one line per epoch",
    )
    parser.add_argument(
        "-m",
        "--MODE",
        choices=["binary", "multi-label"],
        type=str,
        default="multi-label",
        help="binary - performs a binary classification, "
        + "multi-label - performs multi-label classification",
    )
    parser.add_argument(
        "-y",
        "--SET-KIND",
        choices=["all", "testing", "discover"],
        type=str,
        default="all",
        help="all - generates test, train, and validation sets; "
        + "testing - generates a test data set with corresponding labels; "
        + "discover - generates a test data set without labels",
    )
    parser.add_argument(
        "-p", "--STOP-PATIENCE", type=int, default=25, help="Early stopping epoch"
    )
    parser.add_argument(
        "-a",
        "--AUGMENT",
        type=_recognise_boolean,
        default=False,
        help="Choose whether to augment training images or not",
    )
    parser.add_argument(
        "-t",
        "--TEST-SPLIT",
        type=float,
        default=0.2,
        help="The fraction of images used for testing",
    )
    parser.add_argument(
        "-k",
        "--VAL-SPLIT",
        type=float,
        default=0.1,
        help="The fraction of images to be used for validation",
    )
    parser.add_argument(
        "-r",
        "--LEARN-RATE",
        type=float,
        default=0.001,
        help="The optimizer learning rate",
    )
    parser.add_argument(
        "-i",
        "--IMAGE-SIZE",
        type=int,
        nargs="+",
        default=(424, 424),
        help="The size to reshape the images to during preprocessing",
    )
    parser.add_argument(
        "-d",
        "--IMAGE-DATA-PATH",
        type=str,
        default="path/to/images/",
        help="The subdirectory within which the images are stored",
    )
    parser.add_argument(
        "-f",
        "--FILEPATH",
        type=str,
        default="path/to/all/data"
        help="The path to the directory where all files are stored",
    )
    parser.add_argument(
        "-g",
        "--SEPARATE-PATHS",
        type=_recognise_boolean,
        default=False,
        help="If the table and images are stored in the same directory this should be true \
            otherwise false",
    )

    arguments = parser.parse_args()

    return arguments

def read_discover_arguments():
    """
    Function that reads the arguments for discovery from the command line
    This is a trimmed down version of read_train_arguments
    """

    parser = argparse.ArgumentParser(description="CNN training and testing")
    parser.add_argument(
        "table", type=str, help="The name of the table that contains the data"
    )
    parser.add_argument(
        "model", type=str, help="The name of the model weights to be loaded"
    )
    parser.add_argument(
        "-b",
        "--BATCH-SIZE",
        type=int,
        default=75,
        help="The size of training image batches",
    )
    parser.add_argument(
        "-v",
        "--VERBOSE",
        choices=[0, 1, 2],
        type=int,
        default=2,
        help="0 - no updates, 1 - progress bar, 2 - one line per epoch",
    )
    parser.add_argument(
        "-m",
        "--MODE",
        choices=["binary", "multi-label"],
        type=str,
        default="multi-label",
        help="binary - performs a binary classification, "
        + "multi-label - performs multi-label classification",
    )
    parser.add_argument(
        "-i",
        "--IMAGE-SIZE",
        type=int,
        nargs="+",
        default=(424, 424),
        help="The size to reshape the images to during preprocessing",
    )
    parser.add_argument(
        "-d",
        "--IMAGE-DATA-PATH",
        type=str,
        default="path/to/images/",
        help="The directory within which the images are stored",
    )
    parser.add_argument(
        "-f",
        "--FILEPATH",
        type=str,
        default="path/to/all/data/"
        help="The path to the directory where all files are stored (or table if separate paths)",
    )
    parser.add_argument(
        "-g",
        "--SEPARATE-PATHS",
        type=_recognise_boolean,
        default=False,
        help="If the table and images are stored in the same directory this should be true \
            otherwise false",
    )
    parser.add_argument(
        "-x",
        "--MODEL-LOCATION",
        type=str,
        default="path/to/model/",
        help="The path to the directory where all the model weight files are stored"
    )
    parser.add_argument(
        "-n",
        "--MODEL-NUMBER",
        type=int,
        default=0,
        help="The number of the model to be loaded, in the case of predicting on multiple models"
    )

    arguments = parser.parse_args()

    return arguments
