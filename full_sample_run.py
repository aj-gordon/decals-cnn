"""
Name: full_sample_run.py
Author: aj-gordon

This file is used to run pretrained models of the full DECaLS dataset.

"""

# pylint: disable=wrong-import-position

from datetime import datetime

import tensorflow as tf
print(f"[Info] Tensorflow version: {tf.__version__}")
print(f"[Info] Tensorflow devices: {tf.config.list_physical_devices()}")

from utils import Data, Preprocessor, read_discover_arguments, save_csv
from cnns import DecalsNet

if __name__ == "__main__":
    # initialise the model with the configs from comand line and read the arguments
    args = read_discover_arguments()

    # initialise preprocessor model
    processor = Preprocessor(args.IMAGE_SIZE)

    # initialise the data object and load in the data
    data = Data(
        args.table,
        imagepath=args.IMAGE_DATA_PATH,
        generalpath=args.FILEPATH,
        preprocessor=processor,
        set_kind="discover",
        separate_paths=args.SEPARATE_PATHS,
    )
    data.load_data(
        batch_size=args.BATCH_SIZE,
        augment=False,
        mode=args.MODE,
    )

    # get the model and load weights
    if args.MODE == "multi-label":
        model = DecalsNet(
            (args.IMAGE_SIZE[0], args.IMAGE_SIZE[0], 3), output_nodes=4
        ).model
    elif args.MODE == "binary":
        model = DecalsNet(
            (args.IMAGE_SIZE[0], args.IMAGE_SIZE[0], 3), output_nodes=1
        ).model
    else:
        raise ValueError("MODE must be either 'multi-label' or 'binary'")

    weights_file = args.MODEL_LOCATION + args.model
    print(f"[Info] Loading weights from {weights_file}")
    model.load_weights(weights_file)

    # timestamp saves and specify the save location
    now = datetime.now()
    timestamp = now.strftime("%Y_%m_%d-%H_%M_%S")
    name = args.table.split("/")[-1].split(".csv")[0]
    SAVE_LOCATION = f"./outputs/discovery/model{args.MODEL_NUMBER}"

    # get predictions
    predictions = model.predict(data.test_data, verbose=args.VERBOSE)

    # write predictions and labels out to a file
    print("[Info] Saving predictions to csv")
    save_csv(
        predictions,
        data.test_data_names,
        f"{SAVE_LOCATION}/discovery_output-{timestamp}-{name}.csv",
        order=["name", "arm_prediction", "stream_prediction", "shell_prediction",
               "diffuse_prediction"],
        mode=args.MODE,
        set_kind="discover",
    )
