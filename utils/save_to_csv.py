"""
Name: save_to_csv.py
Author: aj-gordon
"""

import pandas as pd
import numpy as np


def _remove_filepath(full_image: str) -> str:
    """Function to remove the filepath from the image name - internal use only"""

    # adds the filepath to the image name
    return full_image.split("/")[-1].split(".png")[0]


def save_csv(
    preds: np.ndarray,
    names: np.ndarray,
    save_loc: str,
    labs: np.ndarray = None,
    cols: tuple = ("arm", "stream", "shell", "diffuse"),
    order: tuple = (
        "name",
        "arm_label",
        "stream_label",
        "shell_label",
        "diffuse_label",
        "arm_prediction",
        "stream_prediction",
        "shell_prediction",
        "diffuse_prediction",
    ),
    mode: str ="multi-label",
    set_kind: str ="test",
):
    """
    Save labels and predictions to a csv file

    Inputs:
        preds (np.ndarray) - 
        names (np.ndarray) -
        save_loc -
        labs -
        cols -
        order -
        mode - 
        set_kind - 
    """
    data_frame = pd.DataFrame()

    _remove_filepath_vec = np.vectorize(_remove_filepath)

    data_frame["name"] = _remove_filepath_vec(names)

    if mode == "multi-label":
        for i, column in enumerate(cols):
            data_frame[f"{column}_prediction"] = preds[:, i]
            if set_kind == "test" or set_kind == "all":
                data_frame[f"{column}_label"] = labs[:, i]
        data_frame = data_frame[order]
    elif mode == "binary":
        if set_kind == "test" or set_kind == "all":
            data_frame["label"] = labs
        data_frame["prediction"] = preds
    else:
        raise ValueError("MODE must be either 'multi-label' or 'binary'")

    data_frame.to_csv(save_loc, sep=",", index=False)
