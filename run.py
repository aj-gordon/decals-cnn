""""
Name: run.py
Author: aj-gordon

"""
# pylint: disable=import-error, wrong-import-position

# import time to save files with a time stamp
from datetime import datetime

# pandas import for saving files
import pandas as pd

# tensorflow imports
import tensorflow as tf

print(f"[Info] Tensorflow version = {tf.version.VERSION}")
print(f"[Info] Tensorflow devices {tf.config.list_physical_devices()}")

# model compilation imports
from tensorflow.keras.losses import BinaryCrossentropy  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore

# file imports
from utils import Data, Preprocessor, read_train_arguments, save_csv
from cnns import DecalsNet

if __name__ == "__main__":
    # initialise the model with the configs from comand line and read the arguments
    args = read_train_arguments()

    # get number of steps per epoch
    nb_steps_per_epoch = args.TRAIN_SIZE // args.BATCH_SIZE

    # initialise preprocessor model
    processor = Preprocessor(args.IMAGE_SIZE)

    # initialise the data object and load in the data
    data = Data(
        args.table,
        imagepath=args.IMAGE_DATA_PATH,
        generalpath=args.FILEPATH,
        preprocessor=processor,
        set_kind=args.SET_KIND,
    )
    data.load_data(
        batch_size=args.BATCH_SIZE,
        augment=args.AUGMENT,
        validation_split=args.VAL_SPLIT,
        test_split=args.TEST_SPLIT,
        mode=args.MODE,
    )

    # define the compiler arguments for the model
    optimiser = Adam(learning_rate=args.LEARN_RATE)
    loss_object = BinaryCrossentropy()
    early_stop = EarlyStopping(
        monitor="val_loss", patience=args.STOP_PATIENCE, restore_best_weights=True
    )

    # get the model and compile

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
    model.compile(optimizer=optimiser, loss=loss_object, metrics=["accuracy"])

    # train the model
    hist = model.fit(
        data.train_data,
        epochs=args.EPOCHS,
        verbose=args.VERBOSE,
        callbacks=[early_stop],
        validation_data=data.val_data,
        class_weight=data.class_weights,
        steps_per_epoch=nb_steps_per_epoch,
    )

    # timestamp saves and specify the save location
    now = datetime.now()
    timestamp = now.strftime("%Y_%m_%d-%H_%M_%S")
    save_name = args.table.split("/")[-1].split("_")[0]
    SAVE_LOCATION = "."

    model.save_weights(f"{SAVE_LOCATION}/models/model_save-{timestamp}-{save_name}")

    # save the hist of the model
    pd.DataFrame(
        {
            "loss": hist.history["loss"],
            "val_loss": hist.history["val_loss"],
            "accuracy": hist.history["accuracy"],
            "val_accuracy": hist.history["val_accuracy"],
        }
    ).to_csv(f"{SAVE_LOCATION}/model_history-{timestamp}-{save_name}.csv", sep=",")

    # get predictions
    predictions = model.predict(data.test_data, verbose=args.VERBOSE)

    # write predictions and labels out to a file
    save_csv(
        predictions,
        data.test_data_names,
        f"{SAVE_LOCATION}/model_output-{timestamp}-{save_name}.csv",
        labs=data.test_data_labels,
        mode=args.MODE
    )
