""""
Name: metrics.py
Author: aj-gordon
"""

# sys import
import sys

# import necessary python packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap

# scikit learn import
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
)  # pylint: disable=wrong-import-position

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Times"

class Metrics(object):
    """Class for determining metrics for a set of predictions and labels"""

    def __init__(self, predictions, labels, name: str, classes: list):
        """
        __init__ constructor for Metrics object

        Inputs:
            predictions (array) - array of classifier predictions
            labels (array) - array of true labels (same shape as predictions)
            name (str) - name to be assigned to the dataset
            classes (list of str) - the names classes being predicted

        Instance variables:
            predictions - the array of predicitions
            labels - the array of labels
            num_classes - the number of classes that are involved
            dataset_name - the name assigned to the dataset
            classes - the names of the classes
        """

        # check predictions and labels are the same shape
        if predictions.shape != labels.shape:
            sys.exit(
                f"Error: predictions {predictions.shape} must have same shape as labels \
                {labels.shape} "
            )

        # set up instance variables
        self.predictions = predictions
        self.labels = labels
        self.num_classes = self.labels.shape[-1]
        self.dataset_name = name
        self.classes = classes

        self.auc = dict()
        self.roc = dict()

        # check if number of classes matches the list of classes
        if len(self.classes) != self.num_classes:
            sys.exit(
                f"Error: list of class names ({len(self.classes)}) must have same number of \
                classes as labels and predictions ({self.num_classes})."
            )

    def roc_and_auc(self):
        """
        Function to calculate the ROC and AUC scores.

        Inputs:
            none

        Instance variables (and outputs):
            roc - the ROC scores for the predictions
            auc - the AUC score for the predictions
        """

        # set dictionaries for fpr, tpr, and auc
        fpr = dict()
        tpr = dict()

        # compute macro roc and auc for each class
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(self.labels[:, i], self.predictions[:, i])
            self.auc[i] = auc(fpr[i], tpr[i])

        # compute micro roc and auc
        fpr["micro"], tpr["micro"], thresh = roc_curve(
            self.labels.ravel(), self.predictions.ravel()
        )
        self.auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # return the values
        self.roc["fpr"] = fpr
        self.roc["tpr"] = tpr

        return self.roc, self.auc, thresh

    ### vectorised function ###
    def from_categorical(self, categorical: list) -> int:
        """
        Function to convert from one-hot encoding into a integer number representative of the class

        Not intended for standalone use

        Inputs:
            categorical (list) - one-hot encoded labels

        Outputs:
            label (int) - integer representation of the class

        """

        # all 0 means no feature
        if np.all(categorical == 0):
            label = 0

        # all 1 means all features
        elif np.all(categorical == 1):
            label = 15

        # if only one feature take index of feature
        elif np.sum(categorical) == 1:
            label = np.argmax(categorical) + 1

        # if two features transform to a number >=5 and <=10
        elif np.sum(categorical) == 2:
            label = np.sum(np.argwhere(categorical == 1)) - categorical[0] + 5

        # if three features transform to a number >= 11 and <= 14
        elif np.sum(categorical) == 3:
            label = np.sum(np.argwhere(categorical == 1)) + 8

        # quit if more than 4 labels
        else:
            sys.exit("Error: cannot understand labels")

        return label

    def confusion_matrices(
        self,
        threshold: float,
        save_location: str,
        colourmap: str = "Blues",
        show: bool = True,
        plot_type: str = "class-wise",
        title: bool = False,
    ):
        """
        Function to plot confusion matrices for the predictions

        Inputs:
            threshold (float) - the prediction threshold to consider as a detection or not
            save_location (str) - the name of the file to save the figure to
            colourmap [opt] (str) - the colourmap to use in plotting
            show [opt] (bool) - choose whether to show the plot or just save
            plot_type [opt] ('class-wise' or 'full') - choose the kind of confusion matrix to plot
            title [opt] (bool) - choose whether the plot has a title or not

        Outputs:
            a plot of the confusion matrix or matrices for the prediction threshold

        """

        if plot_type == "class-wise":
            # set parameters
            number_var = dict(fontsize="large", ha="center", va="bottom")
            norm_var = dict(fontsize="medium", ha="center", va="top")

            # make subplots
            fig, axs = plt.subplots(int(np.ceil(self.num_classes / 2)), 2)

            # add confusion matrices to subplots
            for i in range(self.num_classes):
                cmatrix = np.flip(
                    confusion_matrix(
                        self.labels[:, i],
                        np.where(self.predictions[:, i] >= threshold, 1, 0),
                        normalize=None,
                    ),
                    (1, 0),
                )
                norm_cmatrix = np.flip(
                    confusion_matrix(
                        self.labels[:, i],
                        np.where(self.predictions[:, i] >= threshold, 1, 0),
                        normalize="true",
                    ),
                    (1, 0),
                )

                # plot
                axs[i // 2, i % 2].imshow(cmatrix, cmap=colourmap)

                # change ticks to be class names
                axs[i // 2, i % 2].set_xticks(
                    range(2),
                    labels=[
                        f"{self.classes[i]}",
                        f'No {self.classes[i].split(" ")[-1]}',
                    ],
                    fontsize="small",
                )
                axs[i // 2, i % 2].set_yticks(
                    range(2),
                    labels=[
                        f"{self.classes[i]}",
                        f'No {self.classes[i].split(" ")[-1]}',
                    ],
                    fontsize="small",
                )

                for j in range(2):
                    for k in range(2):
                        if cmatrix[j, k] > 0.5 * (np.amax(cmatrix) + np.amin(cmatrix)):
                            colour = "white"
                        else:
                            colour = "black"

                        # add the number to the plot
                        axs[i // 2, i % 2].text(
                            k, j, f"{cmatrix[j,k]}", color=colour, **number_var
                        )
                        axs[i // 2, i % 2].text(
                            k, j, f"({norm_cmatrix[j,k]:.2%})", color=colour, **norm_var
                        )

                axs[i // 2, i % 2].set_title(f"{self.classes[i]}")

            # make it look nice
            if self.num_classes % 2 != 0:
                axs[-1, -1].axis("off")

            fig.tight_layout()
            fig.supxlabel("Predicted Labels")
            fig.supylabel("True Labels")

            # add title
            if title:
                fig.suptitle("Class-wise Confusion Matrices")

            # save
            plt.savefig(
                f'{save_location.split(".")[0]}_classwise.{save_location.split(".")[-1]}'
            )

        elif plot_type == "full":
            # set parameters
            number_var = dict(fontsize="small", ha="center", va="center")

            # convert one-hot labels and predictions to seperate classes
            labels = np.apply_along_axis(self.from_categorical, -1, self.labels)
            predictions = np.apply_along_axis(
                self.from_categorical, -1, np.where(self.predictions >= threshold, 1, 0)
            )

            # generate confusion matrices
            cmatrix = np.roll(
                confusion_matrix(labels, predictions, normalize=None), 1, axis=(0, 1)
            )
            norm_cmatrix = np.roll(
                confusion_matrix(labels, predictions, normalize="true"), 1, axis=(0, 1)
            )

            # make plot
            fig = plt.figure()

            # plot
            plt.imshow(cmatrix, cmap=colourmap)

            # change ticks to class names
            names = [[x for x in self.classes]]
            names.append(
                [
                    rf"{self.classes[i].split(' ')[-1]} \& {self.classes[j].split(' ')[-1]}"
                    for i in range(self.num_classes)
                    for j in range(i + 1, self.num_classes)
                ]
            )
            names.append(
                [
                    rf"{self.classes[i].split(' ')[-1]}, {self.classes[j].split(' ')[-1]} \
                \& {self.classes[k].split(' ')[-1]}"
                    for i in range(self.num_classes)
                    for j in range(i + 1, self.num_classes)
                    for k in range(j + 1, self.num_classes)
                ]
            )
            names.append(["All"])
            names.append(["None"])
            names = [element for row in names for element in row]

            plt.xticks(range(len(names)), labels=names, rotation=90, fontsize="x-small")
            plt.yticks(range(len(names)), labels=names, fontsize="x-small")

            for j in range(len(names)):
                for k in range(len(names)):
                    if cmatrix[j, k] > 0.5 * (np.amax(cmatrix) + np.amin(cmatrix)):
                        colour = "white"
                    else:
                        colour = "black"

                    # add number to plot
                    plt.text(k, j, f"{cmatrix[j,k]}", color=colour, **number_var)

            # make it look nice
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")

            # add title
            if title:
                plt.title("Class-wise Confusion Matrices")

            plt.tight_layout()

            # save
            plt.savefig(
                f'{save_location.split(".")[0]}_full.{save_location.split(".")[-1]}'
            )

        else:
            # close if wrong plot_type provided
            sys.exit(
                "Error: the specified plot_type must be either 'class-wise' or 'full'."
            )

        # show if showing otherwise close
        if show:
            plt.show()
        else:
            plt.close()

    def precision(self, threshold: float, average_type: str = "micro"):
        """
        Function to calculate the averaged or class-wise precision score

        Inputs:
            threshold (float) - the prediction threshold to consider as a detection or not
            average_type [opt] (str) - choose the averaged or class-wise precision score

        Outputs:
            [averaged] precision_score (float) - the averaged precision score
            [class-wise] preicision_scores (dict) - the precision scores for each class
        """

        # averaged precision
        if average_type in ["micro", "macro", "samples", "weighted", "binary"]:
            return precision_score(
                self.labels,
                np.where(self.predictions >= threshold, 1, 0),
                average=average_type,
            )

        # class-wise precision
        elif average_type == "class":
            return dict(
                zip(
                    self.classes,
                    precision_score(
                        self.labels,
                        np.where(self.predictions >= threshold, 1, 0),
                        average=None,
                    ),
                )
            )

        else:
            # quit if wrong average or not class-wise
            sys.exit(
                "Error: invalid average_type passed to precision function, select from 'micro'\
                , 'macro', 'samples', 'weighted', or 'binary' for averaged recall or 'class' for \
                class-wise."
            )

    def accuracy(self, threshold: float, average_type: str = "subset"):
        """
        Function to calculate the averaged or subset accuracy score

        Inputs:
            threshold (float) - the prediction threshold to consider as a detection or not
            average_type [opt] ('subset' or 'class') - choose the element-wise or class-wise \
                accuracy score

        Outputs:
            [subset] accuracy_score (float) - the subset accuracy score
            [class-wise] accuracy_scores (dict) - the accuracy scores for each class
        """

        # subset accuracy
        if average_type == "subset":
            return accuracy_score(
                self.labels, np.where(self.predictions >= threshold, 1, 0)
            )

        # class-wise accuracy
        elif average_type == "class":
            accuracy = {}

            for i in range(self.num_classes):
                accuracy[self.classes[i]] = accuracy_score(
                    self.labels[:, i],
                    np.where(self.predictions[:, i] >= threshold, 1, 0),
                )

            return accuracy

        else:
            # quit if wrong average_type
            sys.exit(
                "Error: invalid average_type passed to accuracy function, select from 'subset'\
                      for sample-wise or 'class' class-wise."
            )

    def recall(self, threshold: float, average_type: str = "micro"):
        """
        Function to calculate the averaged or class-wise recall score

        Inputs:
            threshold (float) - the prediction threshold to consider as a detection or not
            average_type [opt] (str) - choose the average or class-wise recall score

        Outputs:
            [average] recall_score (float) - the averaged recall score
            [class-wise] recall_scores (dict) - the recall scores for each class
        """

        # averaged recall
        if average_type in ["micro", "macro", "samples", "weighted", "binary"]:
            return recall_score(
                self.labels,
                np.where(self.predictions >= threshold, 1, 0),
                average=average_type,
            )

        # class-wise recall
        elif average_type == "class":
            return dict(
                zip(
                    self.classes,
                    recall_score(
                        self.labels,
                        np.where(self.predictions >= threshold, 1, 0),
                        average=None,
                    ),
                )
            )

        else:
            # quit if wrong average_type
            sys.exit(
                "Error: invalid average_type passed to recall function, select from 'micro', \
                     'macro', 'samples', 'weighted', or 'binary' for averaged recall or 'class' \
                     for class-wise."
            )

    def get_f1(self, threshold: float, average_type: str = "micro"):
        """
        Function to calculate the averaged or class-wise F1 score

        Inputs:
            threshold (float) - the prediction threshold to consider as a detection or not
            average_type [opt] (str) - choose the average or class-wise F1 score

        Outputs:
            [average] f1_score (float) - the averaged F1 score
            [class-wise] f1_scores (dict) - the F1 scores for each class
        """

        # averaged f1 score
        if average_type in ["micro", "macro", "samples", "weighted", "binary"]:
            return f1_score(
                self.labels,
                np.where(self.predictions >= threshold, 1, 0),
                average=average_type,
            )

        # class-wise f1 score
        elif average_type == "class":
            return dict(
                zip(
                    self.classes,
                    f1_score(
                        self.labels,
                        np.where(self.predictions >= threshold, 1, 0),
                        average=None,
                    ),
                )
            )

        else:
            # quit if wrong average_type
            sys.exit(
                "Error: invalid average_type passed to f1 function, select from 'micro', \
                     'macro', 'samples', 'weighted', or 'binary' for averaged recall or 'class' \
                     for class-wise."
            )

    @classmethod
    def plot_roc(
        cls,
        metric_objects: list,
        save_location: str,
        colourmap: str = "plasma",
        show: bool = True,
        plot_type: str = "micro",
        title: bool = False,
    ):
        """
        Function to plot ROC curves for many instances of the Metric class

        Inputs:
            metric_objects (list) - the instances of the metric class to plot ROC curves for
            save_location (str) - the name of the file to save the figure to
            colourmap [opt] (str) - the colourmap to use in plotting
            show [opt] (bool) - choose whether to show the plot or just save
            plot_type [opt] ('micro' or 'macro') - choose the kind of ROC curves to plot
            title [opt] (bool) - choose whether the plot has a title or not

        Outputs:
            [micro] a single plot of the ROC curves for all metric objects
            [macro] a plot of the ROC curves for the classes in each metric object
        """

        # get colourmap
        cmap = get_cmap(colourmap)

        # plot micro average roc curves for a series of metric objects
        if plot_type == "micro":
            # normalize item number values to colourmap
            norm = Normalize(vmin=0, vmax=len(metric_objects) + 1)

            for i, metric_object in enumerate(metric_objects):
                # get roc and auc
                roc, auc_val, _ = metric_object.roc_and_auc()

                # one plot for all metric objects
                plt.plot(
                    roc["fpr"]["micro"],
                    roc["tpr"]["micro"],
                    lw=2,
                    color=cmap(norm(i + 1)),
                    label=f'{metric_object.dataset_name} \
                            (area = {auc_val["micro"]:.2})',
                )

            # add random guessing
            plt.plot(
                [0, 1],
                [0, 1],
                color=cmap(norm(0)),
                lw=2,
                linestyle="--",
                label="Random Guessing",
            )

            # make it look nice
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")

            # add title
            if title:
                plt.title("Receiver Operating Curve")

            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.minorticks_on()
            plt.tick_params(direction='in', which='both')

            # save
            plt.savefig(save_location)

            # show
            if show:
                plt.show()
            else:
                plt.close()

        # plot the class macro roc curves for a single metric object
        elif plot_type == "macro":
            # one plot for each object
            for i, metric_object in enumerate(metric_objects):
                # normalize class number values to colourmap
                norm = Normalize(vmin=0, vmax=metric_object.num_classes)

                # get roc and auc
                roc, auc_val, _ = metric_object.roc_and_auc()

                # plot
                for j in range(metric_object.num_classes):
                    plt.plot(
                        roc["fpr"][j],
                        roc["tpr"][j],
                        lw=2,
                        color=cmap(norm(j + 1)),
                        label=f"{metric_object.classes[j]} \
                                (area = {auc_val[j]:.2})",
                    )

                # add random guessing
                plt.plot(
                    [0, 1],
                    [0, 1],
                    color=cmap(norm(0)),
                    lw=2,
                    linestyle="--",
                    label="Random Guessing",
                )

                # make it look nice
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.0])
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")

                # add title
                if title:
                    plt.title(f"Receiver Operating Curve {metric_object.dataset_name}")

                plt.legend(loc="lower right")
                plt.tight_layout()
                plt.minorticks_on()
                plt.tick_params(direction='in', which='both')

                # save
                plt.savefig(
                    f"{save_location.split('.')[0]}_{metric_object.dataset_name}"
                    + f".{save_location.split('.')[-1]}"
                )

                # show
                if show:
                    plt.show()
                else:
                    plt.close()

        # quit if not selected micro or macro
        else:
            sys.exit(
                "Error: the specified plot_type must be either 'micro' or 'macro'."
            )


if __name__ == "__main__":
    # don't run the code as a main program
    sys.exit("Error: this code is not intended to be run as a main file.")
