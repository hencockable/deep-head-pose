
from my_scripts import my_utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import numpy as np

if __name__ == '__main__':
    PATH_TO_ANNOTATIONS1 = "../../source/annotations_0925_G12_Chemistry_cut_VP3.txt"
    PATH_TO_ANNOTATIONS2 = "../../source/annotations_0925_G12_Chemistry_cut_VP2.txt"
    N_SAMPLES_PL = 15  # samples per label
    K_CROSS_VAL = 10     # number of train/test per subset of the entire data
    RUNS = 1         # number of runs of cross val with different data subsets
    TEST_SIZE = 0.33    # percentage of data set to use for testing

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
            X1, y1 = my_utils.prepare_annotations(PATH_TO_ANNOTATIONS1, n=N_SAMPLES_PL)
            X2, y2 = my_utils.prepare_annotations(PATH_TO_ANNOTATIONS2, n=N_SAMPLES_PL)

            # train and test classifier with training on one data set and testing on the other
            accuracy, recall, precision, roc_auc = my_utils.train_and_test_different_data(clf, X1, y1, X2, y2, K_CROSS_VAL, test_size=TEST_SIZE)

            total_accuracies.append(accuracy)
            total_recalls.append(recall)
            total_precisions.append(precision)
            total_roc_aucs.append(roc_auc)

        # print statistics (accuracy with 95% confidence intervall - normal distribution assumed)
        out = "\n{}:\n\nAccuracy: {:0.2f} (+/- {:0.2f})\nRecall: \t\t{:.2f}\nPrecision: " \
              "\t\t{:.2f}\nROC-AUC: \t\t{:.2f}\n".format(name, np.mean(total_accuracies), np.std(total_accuracies) * 2, np.mean(total_recalls), np.mean(total_precisions), np.mean(total_roc_aucs))

        divider = "-" * 35

        print(divider, out)
