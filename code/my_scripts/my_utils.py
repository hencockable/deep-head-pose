import seaborn as sb
from scipy import stats
from random import shuffle
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import recall_score, precision_score, roc_auc_score, plot_confusion_matrix, accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.base import clone
import matplotlib.pyplot as plt
import numpy as np


def get_features_and_labels(annotations):
    X = annotations[["yaw", "pitch", "roll"]].values.tolist()
    y = annotations["label"].values.tolist()

    return X, y


def show_label_dist(annotations):
    sb.distplot(annotations["label"], kde=False, bins=10)
    plt.ylabel("n(label)")
    plt.xlabel("Labels")
    plt.show()


def uniform_distribution(annotations, n=None):
    # get unique labels
    labels = annotations.label.unique()

    label_dict = {}
    sizes = []

    for label in labels:
        # group samples by label
        label_dict[label] = annotations.loc[annotations["label"] == label]
        sizes.append(len(label_dict[label]))

    minimum = min(sizes)

    if n is None or n > minimum:
        print("N_SAMPLES_PL larger than number of least frequent label -> N_SAMPLES_PL "
              "set to number of least frequent label.")
        n = minimum

    data = []

    for label in labels:
        records = label_dict[label].sample(n).to_records(index=False)
        data.extend(list(records))

    shuffle(data)

    return pd.DataFrame.from_records(data, columns=["yaw", "pitch", "roll", "label"])


def one_hot_encoder(labels):
    uniques = list(set(labels))
    out = []

    for label in labels:
        current = [0 for i in range(len(uniques))]
        current[uniques.index(label)] = 1
        out.append(current)

    return uniques, out


def prepare_annotations(path, n=None):
    # read annotations
    annotations = pd.read_csv(path, delimiter=" ")

    # exclude label 4 (other) since its not relevant for synchrony
    annotations = annotations[annotations.label != 4]

    # show initial distribution of labels
    # my_utils.show_label_dist(annotations)

    # make labels uniformly distributed
    annotations = uniform_distribution(annotations, n=n)

    # show distribution after uniform
    # my_utils.show_label_dist(annotations)

    # split annotations into features and labels and return (X, y)
    return get_features_and_labels(annotations)


def train_and_test_different_data(clf, X1, y1, X2, y2, k_cross_val, test_size):
    accuracies = []
    recalls = []
    precisions = []
    roc_aucs = []

    for i in range(k_cross_val):
        print("K = {}".format(i))
        # clone classifier to get settings but discard previous trainings
        copy = clone(clf)

        # create train and test sets (1 used for training, 2 used for testing)
        X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=test_size, random_state=i,
                                                                shuffle=True, stratify=y1)
        X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=test_size, random_state=i,
                                                                shuffle=True, stratify=y2)
        X_train, y_train = X1_train, y1_train
        X_test, y_test = X2_test, y2_test

        # convert labels to one-hot representation for multi class roc-auc
        y_test_bin = label_binarize(y2_test, classes=[0, 1, 2, 3])

        # train classifier
        copy.fit(X_train, y_train)

        # get test predictions
        preds = copy.predict(X_test)

        # get certainties/probabilities per label per prediction
        try:
            probs = copy.predict_proba(X_test)
        except AttributeError:
            probs = label_binarize(preds, classes=[0, 1, 2, 3])

        # compute accuracy score
        accuracies.append(accuracy_score(y_test, preds))

        # compute recall = tp / (tp + fn)
        recalls.append(recall_score(y_test, preds, average="weighted"))

        # compute precision = tp / (tp + fp)
        precisions.append(precision_score(y_test, preds, average="weighted"))

        # compute roc-auc score (y: tp, x: fp)
        roc_aucs.append(roc_auc_score(y_test_bin, probs, average="weighted", multi_class="ovr"))

        # # compute and plot confusion matrix
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
        # ax = plt.gca()
        # plot_confusion_matrix(clf, X_test, y_test, ax=ax, values_format=".2f", normalize="true")
        # plt.title("Confusion Matrix {}".format(name))
        # plt.show()

    return np.mean(accuracies), np.mean(recalls), np.mean(precisions), np.mean(roc_aucs)


def train_and_test_same_data(clf, X, y, k_cross_val, test_size=0.33):
    accuracies = []
    recalls = []
    precisions = []
    roc_aucs = []

    for i in range(k_cross_val):
        print("K = {}".format(i))
        # clone classifier to get settings but discard previous trainings
        copy = clone(clf)

        # create train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i, shuffle=True, stratify=y)

        # convert labels to one-hot representation for multi class roc-auc
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])

        # train classifier
        copy.fit(X_train, y_train)

        # get test predictions
        preds = copy.predict(X_test)

        # get certainties/probabilities per label per prediction
        try:
            probs = copy.predict_proba(X_test)
        except AttributeError:
            probs = label_binarize(preds, classes=[0, 1, 2, 3])

        # compute accuracy score
        accuracies.append(accuracy_score(y_test, preds))

        # compute recall = tp / (tp + fn)
        recalls.append(recall_score(y_test, preds, average="weighted"))

        # compute precision = tp / (tp + fp)
        precisions.append(precision_score(y_test, preds, average="weighted"))

        # compute roc-auc score (y: tp, x: fp)
        roc_aucs.append(roc_auc_score(y_test_bin, probs, average="weighted", multi_class="ovr"))

    # # compute and plot confusion matrix
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
    # ax = plt.gca()
    # plot_confusion_matrix(clf, X_test, y_test, ax=ax, values_format=".2f", normalize="true")
    # plt.title("Confusion Matrix {}".format(name))
    # plt.show()

    return np.mean(accuracies), np.mean(recalls), np.mean(precisions), np.mean(roc_aucs)
