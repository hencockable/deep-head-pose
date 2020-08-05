from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix
import pandas as pd
from random import shuffle
import matplotlib.pyplot as plt
import seaborn as sb
import os
import joblib

sb.set(style="whitegrid")


def sub_sample(X_train, y_train, sub_sample_size):
    # re-join X_train and y_train for more convenient sub-sampling per label
    joined_train = pd.DataFrame({"X_train": X_train, "y_train": y_train})

    # get unique labels
    labels = joined_train.y_train.unique()

    label_dict = {}
    sizes = []

    for label in labels:
        # group samples by label
        label_dict[label] = joined_train.loc[joined_train["y_train"] == label]
        sizes.append(len(label_dict[label]))

    minimum = min(sizes)

    if sub_sample_size > minimum:
        print("sub_sample_size larger than number of least frequent label -> sub_sample_size "
              "set to number of least frequent label: {}.".format(minimum))
        sub_sample_size = minimum

    data = []

    for label in labels:
        records = label_dict[label].sample(sub_sample_size).to_records(index=False)
        data.extend(list(records))

    shuffle(data)

    sub_sampled_train_set = pd.DataFrame.from_records(data, columns=["X_train", "y_train"])

    X_train_sub = sub_sampled_train_set.X_train.to_list()
    y_train_sub = sub_sampled_train_set.y_train.to_list()

    return X_train_sub, y_train_sub


PATH_TO_ANNOTATIONS = "/home/hendrik/PycharmProjects/Master/Hopenet/source/annotations_0927_G08_French_cut/perSecond/"
RUNS = 10  # number of runs of cross val with different data subsets
TEST_SIZE = 0.2  # percentage of data set to use for testing
LABELS = [0, 1, 2]
sub_samples_per_label = ["all", 17, 15, 10, 5]

for file in os.listdir(PATH_TO_ANNOTATIONS):
    print("Current annotations to process: {}".format(file))
    SAVE_PATH_PLOTS = "../output/plots/" + file + "_joined_l2l3/"
    SAVE_PATH_CLF = "../output/clfs/" + file + "_joined_l2l3/"

    # check if save dir plots exists, else create it
    if not os.path.exists(SAVE_PATH_PLOTS):
        os.makedirs(SAVE_PATH_PLOTS)
        print("Created dir: {}".format(SAVE_PATH_PLOTS))
    else:
        print("Directory already exists: {}".format(SAVE_PATH_PLOTS))

    # check if save dir clf exists, else create it
    if not os.path.exists(SAVE_PATH_CLF):
        os.makedirs(SAVE_PATH_CLF)
        print("Created dir: {}".format(SAVE_PATH_CLF))
    else:
        print("Directory already exists: {}".format(SAVE_PATH_CLF))

    # create test and train set
    # read annotations
    annotations = pd.read_csv(PATH_TO_ANNOTATIONS + file)

    # exclude label 4 if present (other) since its not relevant for synchrony
    annotations = annotations[annotations.label != 4]

    # split annotations into features and labels (X, y)
    X = annotations[["yaw", "pitch", "roll"]].values.tolist()
    y = annotations["label"].astype("int32").values.tolist()


    # Pool labels 2 and 3 because of little sample size
    y = [2 if i == 3 else i for i in y]

    # make train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42, shuffle=True,
                                                        stratify=y)


    # # plot train label distribution
    sb.catplot(x="y_train", kind="count", palette=sb.cubehelix_palette(n_colors=4), data=pd.DataFrame(y_train, columns=["y_train"]))
    plt.xticks(LABELS, ["Front ({})".format(y_train.count(0)),
                        "Desk ({})".format(y_train.count(1)),
                        "Left/Right ({})".format(y_train.count(2))])  # , "Right ({})".format(y_train.count(3))])
    plt.title("Label Distribution Training Set")
    plt.ylabel("#samples")
    plt.xlabel("Labels")
    plt.savefig(SAVE_PATH_PLOTS + "label_dist.png")
    plt.show()


    # initialize classifiers
    classifiers = [SVC(kernel="rbf")]

    for clf in classifiers:
        # get name
        name = clf.__class__.__name__

        # create Dataframe to save results in
        df = pd.DataFrame(columns=["sub_sample_size", "run", "accuracy", "precision", "recall"])

        # iterate over the different subsample sizes
        for sub_sample_size in sub_samples_per_label:

            total_preds = []
            total_truth = []

            # train and test RUNS times for each subset
            for run in range(RUNS):

                # clone classifier to get settings but discard previous trainings
                copy = clone(clf)

                if sub_sample_size is "all":
                    X_train_sub, y_train_sub = X_train, y_train

                else:
                    X_train_sub, y_train_sub = sub_sample(X_train, y_train, sub_sample_size)

                # train classifier
                copy.fit(X_train_sub, y_train_sub)

                # save first classifier
                if run == 0:
                    joblib.dump(copy, "{}{}_{}.sav".format(SAVE_PATH_CLF, name, sub_sample_size))

                # get test predictions
                preds = copy.predict(X_test)

                # save preds and ground truth for confusion matrix
                total_preds.extend(preds)
                total_truth.extend(y_test)

                # compute accuracy score
                accuracy = accuracy_score(y_test, preds)

                # compute recall = tp / (tp + fn)
                recall = recall_score(y_test, preds, average="macro")

                # compute precision = tp / (tp + fp)
                precision = precision_score(y_test, preds, average="macro")

                df.loc[len(df)] = [sub_sample_size, run, accuracy, precision, recall]

            # plot confusion matrix
            cmat = confusion_matrix(total_truth, total_preds, normalize="true", labels=LABELS)
            df_cmat = pd.DataFrame(cmat, index=LABELS, columns=LABELS)
            sb.heatmap(df_cmat, annot=True, cmap=sb.cubehelix_palette(n_colors=999, dark=0.3))
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            plt.title("{} - Confusion Matrix - {}".format(name, sub_sample_size))
            plt.savefig(fname=SAVE_PATH_PLOTS + "cmat_{}_{}.png".format(name, sub_sample_size))
            plt.show()

        plt.title("{} - Accuracy".format(name))
        ax1 = sb.boxplot(x="sub_sample_size", y="accuracy", data=df, palette=sb.cubehelix_palette(n_colors=9, dark=0.3))
        plt.savefig(fname=SAVE_PATH_PLOTS + "accu_{}.png".format(name))
        plt.show()

        plt.title("{} - Precision".format(name))
        ax2 = sb.boxplot(x="sub_sample_size", y="precision", data=df, palette=sb.cubehelix_palette(n_colors=9, dark=0.3))
        plt.savefig(fname=SAVE_PATH_PLOTS + "pre_{}.png".format(name))
        plt.show()

        plt.title("{} - Recall".format(name))
        ax3 = sb.boxplot(x="sub_sample_size", y="recall", data=df, palette=sb.cubehelix_palette(n_colors=9, dark=0.3))
        plt.savefig(fname=SAVE_PATH_PLOTS + "rec_{}.png".format(name))
        plt.show()
