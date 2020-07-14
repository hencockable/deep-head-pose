from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.metrics import recall_score, precision_score, accuracy_score
import pandas as pd
from random import shuffle
import matplotlib.pyplot as plt
import seaborn as sb

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


PATH_TO_ANNOTATIONS = "~/PycharmProjects/Master/Hopenet/source/annotations_0925_G12_Chemistry_cut_VP2.txt"
RUNS = 10  # number of runs of cross val with different data subsets
TEST_SIZE = 0.2  # percentage of data set to use for testing
sub_samples_per_label = ["all", 100, 50, 40, 30, 20, 15, 10, 5]

# create test and train set
# read annotations
annotations = pd.read_csv(PATH_TO_ANNOTATIONS, delimiter=" ")

# exclude label 4 if present (other) since its not relevant for synchrony
annotations = annotations[annotations.label != 4]

# split annotations into features and labels (X, y)
X = annotations[["yaw", "pitch", "roll"]].values.tolist()
y = annotations["label"].values.tolist()

# make train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42, shuffle=True,
                                                    stratify=y)

# plot train label distribution
sb.catplot(x="y_train", kind="count", palette=sb.cubehelix_palette(n_colors=4), data=pd.DataFrame(y_train, columns=["y_train"]))
plt.xticks([0, 1, 2, 3], ["Front (0)", "Desk (1)", "Left (2)", "Right (3)"])
plt.title("Label Distribution Training Set")
plt.ylabel("#samples")
plt.xlabel("Labels")
plt.show()


# initialize classifiers
classifiers = [RandomForestClassifier(),
               LinearSVC(dual=False),
               LogisticRegression(dual=False, max_iter=2000, multi_class="multinomial")]

for clf in classifiers:
    # create Dataframe to save results in
    df = pd.DataFrame(columns=["sub_sample_size", "run", "accuracy", "precision", "recall"])

    # iterate over the different subsample sizes
    for sub_sample_size in sub_samples_per_label:

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

            # get test predictions
            preds = copy.predict(X_test)

            # compute accuracy score
            accuracy = accuracy_score(y_test, preds)

            # compute recall = tp / (tp + fn)
            recall = recall_score(y_test, preds, average="macro")

            # compute precision = tp / (tp + fp)
            precision = precision_score(y_test, preds, average="macro")

            df.loc[len(df)] = [sub_sample_size, run, accuracy, precision, recall]

    plt.title("{} - Accuracy".format(clf.__class__.__name__))
    ax1 = sb.boxplot(x="sub_sample_size", y="accuracy", data=df, palette=sb.cubehelix_palette(n_colors=9, dark=0.3))
    plt.show()

    plt.title("{} - Precision".format(clf.__class__.__name__))
    ax2 = sb.boxplot(x="sub_sample_size", y="precision", data=df, palette=sb.cubehelix_palette(n_colors=9, dark=0.3))
    plt.show()

    plt.title("{} - Recall".format(clf.__class__.__name__))
    ax3 = sb.boxplot(x="sub_sample_size", y="recall", data=df, palette=sb.cubehelix_palette(n_colors=9, dark=0.3))
    plt.show()
