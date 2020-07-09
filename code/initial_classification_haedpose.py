import my_utils
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import recall_score, precision_score, roc_auc_score, plot_confusion_matrix, accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.base import clone
import matplotlib.pyplot as plt
import numpy as np


def prepare_annotations(path, n=None):
    # read annotations
    annotations = pd.read_csv(path, delimiter=" ")

    # exclude label 4 (other) since its not relevant for synchrony
    annotations = annotations[annotations.label != 4]

    # show initial distribution of labels
    # my_utils.show_label_dist(annotations)

    # make labels uniformly distributed
    annotations = my_utils.uniform_distribution(annotations, n=n)

    # show distribution after uniform
    # my_utils.show_label_dist(annotations)

    # split annotations into features and labels and return (X, y)
    return my_utils.get_features_and_labels(annotations)


def train_and_test(clf, X, y, k_cross_val):
    accuracies = []
    recalls = []
    precisions = []
    roc_aucs = []

    for i in range(k_cross_val):
        # clone classifier to get settings but discard previous trainings
        clf = clone(clf)

        # create train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=i, shuffle=True, stratify=y)

        # convert labels to one-hot representation for multi class roc-auc
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])

        # train classifier
        clf.fit(X_train, y_train)

        # get test predictions
        preds = clf.predict(X_test)

        # get certainties/probabilities per label per prediction
        try:
            probs = clf.predict_proba(X_test)
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


if __name__ == '__main__':
    PATH_TO_ANNOTATIONS = "../source/annotations_0925_G12_Chemistry_cut_VP2.txt"
    N_SAMPLES = 15
    K_CROSS_VAL = 10
    RUNS = 100

    # initialize classifiers
    classifiers = [RandomForestClassifier(),
                   LinearSVC(dual=False),
                   LogisticRegression(dual=False, max_iter=2000)]

    for clf in classifiers:
        # get classifier name
        name = clf.__class__.__name__

        total_accuracies = []
        total_recalls = []
        total_precisions = []
        total_roc_aucs = []

        for run in range(RUNS):
            print(name, run)

            # get features and labels (shuffled)
            X, y = prepare_annotations(PATH_TO_ANNOTATIONS, n=N_SAMPLES)

            # train and test classifier
            accuracy, recall, precision, roc_auc = train_and_test(clf, X, y, K_CROSS_VAL)

            total_accuracies.append(accuracy)
            total_recalls.append(recall)
            total_precisions.append(precision)
            total_roc_aucs.append(roc_auc)

        # print statistics (accuracy with 95% confidence intervall - normal distribution assumed)
        out = "\n{}:\n\nAccuracy: {:0.2f} (+/- {:0.2f})\nRecall: \t\t{:.2f}\nPrecision: " \
              "\t\t{:.2f}\nROC-AUC: \t\t{:.2f}\n".format(name, np.mean(total_accuracies), np.std(total_accuracies) * 2, np.mean(total_recalls), np.mean(total_precisions), np.mean(total_roc_aucs))

        divider = "-" * 35

        print(divider, out)
