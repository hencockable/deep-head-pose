import my_utils
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import recall_score, precision_score, roc_auc_score, plot_confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt


def prepare_annotations(path):
    # read annotations
    annotations = pd.read_csv(path, delimiter=" ")

    # show initial distribution of labels
    my_utils.show_label_dist(annotations)

    # make labels uniformly distributed
    annotations = my_utils.uniform_distribution(annotations)

    # show distribution after uniform
    my_utils.show_label_dist(annotations)

    # split annotations into features and labels and return (X, y)
    return my_utils.get_features_and_labels(annotations)


def train_and_test(clf, X, y):
    # get classifier type
    name = clf.__class__.__name__

    # create train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    # convert labels to one-hot representation for multi class roc-auc
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4])

    # train classifier
    clf.fit(X_train, y_train)

    # compute kfold accuracy
    accuracies = cross_val_score(clf, X, y, cv=10)

    # get test predictions
    preds = clf.predict(X_test)

    # get certainties/probabilities per label per prediction
    try:
        probs = clf.predict_proba(X_test)
    except AttributeError:
        probs = label_binarize(preds, classes=[0, 1, 2, 3, 4])

    # compute recall = tp / (tp + fn)
    recall = recall_score(y_test, preds, average="weighted")

    # compute precision = tp / (tp + fp)
    precision = precision_score(y_test, preds, average="weighted")

    # compute roc-auc score (y: tp, x: fp)
    roc_auc = roc_auc_score(y_test_bin, probs, average="weighted", multi_class="ovr")

    # compute and plot confusion matrix
    ax = plt.gca()
    plot_confusion_matrix(clf, X_test, y_test, ax=ax, values_format=".2f", normalize="true")
    plt.title("Confusion Matrix {}".format(name))
    plt.show()

    # print statistics (accuracy with 95% confidence intervall - normal distribution assumed)
    out = "\n{}:\n\nAccuracy: {:0.2f} (+/- {:0.2f})\nRecall: \t\t{:.2f}\nPrecision: " \
          "\t\t{:.2f}\nROC-AUC: \t\t{:.2f}\n".format(name, accuracies.mean(), accuracies.std() * 2, recall, precision, roc_auc)

    divider = "-" * 35

    print(divider, out)


if __name__ == '__main__':
    PATH_TO_ANNOTATIONS = "../source/annotations_0925_G12_Chemistry_cut_VP2.txt"

    # get features and labels
    X, y = prepare_annotations(PATH_TO_ANNOTATIONS)

    # initialize classifier
    rfc = RandomForestClassifier()
    svc = LinearSVC(dual=False)
    lrc = LogisticRegression(dual=False, max_iter=2000)


    # train and test classifier
    train_and_test(rfc, X, y)
    train_and_test(svc, X, y)
    train_and_test(lrc, X, y)



