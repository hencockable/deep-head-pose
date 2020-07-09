import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from scipy import stats
from random import shuffle
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

    if n is None:
        minimum = min(sizes)
    else:
        minimum = n

    data = []

    for label in labels:
        records = label_dict[label].sample(minimum).to_records(index=False)
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
