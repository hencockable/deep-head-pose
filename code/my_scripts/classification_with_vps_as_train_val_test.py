import os
import pandas as pd
from sklearn.svm import SVC
from random import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sb

# total number vps: 13, train: 9, val: 3, test: 1

data_path = "../../source/train_val_test_sets/"
save_path_plots = "../output/plots/classification_with_vps_as_train_val_test/"

files = os.listdir(data_path + "data/")

LABELS = [0, 1, 2]

out_df = pd.DataFrame([], columns=["test", "val", "train", "hp_val_acc", "l4_val_acc", "hp_test_acc", "l4_test_acc"])

for vp in files:
    test_file = vp
    copy = files.copy()
    copy.remove(test_file)
    shuffle(copy)
    train_files = copy[:9]
    val_files = copy[9:]

    train_hps = []
    val_hps = []
    test_hps = []

    train_l4 = []
    val_l4 = []
    test_l4 = []

    train_labels = []
    val_labels = []
    test_labels = []

    for train_file in train_files:
        hp_df = pd.read_csv("{}data/{}".format(data_path, train_file))
        l4_df = pd.read_csv("{}l4/{}".format(data_path, train_file))

        train_hps.extend(hp_df[["yaw", "pitch", "roll"]].values.tolist())
        train_l4.extend(l4_df.values.tolist())

        train_labels.extend(hp_df["label"].values.tolist())

    for val_file in val_files:
        hp_df = pd.read_csv("{}data/{}".format(data_path, val_file))
        l4_df = pd.read_csv("{}l4/{}".format(data_path, val_file))

        val_hps.extend(hp_df[["yaw", "pitch", "roll"]].values.tolist())
        val_l4.extend(l4_df.values.tolist())

        val_labels.extend(hp_df["label"].values.tolist())

    hp_df = pd.read_csv("{}data/{}".format(data_path, test_file))
    l4_df = pd.read_csv("{}l4/{}".format(data_path, test_file))

    test_hps.extend(hp_df[["yaw", "pitch", "roll"]].values.tolist())
    test_l4.extend(l4_df.values.tolist())

    test_labels.extend(hp_df["label"].values.tolist())

    # pool labels 2 & 3
    train_labels = [2 if i == 3 else int(i) for i in train_labels]
    val_labels = [2 if i == 3 else int(i) for i in val_labels]
    test_labels = [2 if i == 3 else int(i) for i in test_labels]

    # inintialize clfs
    hp_clf = SVC(kernel="rbf")
    l4_clf = SVC(kernel="rbf")

    # train clfs
    hp_clf.fit(train_hps, train_labels)
    l4_clf.fit(train_l4, train_labels)

    # predict
    hp_val_preds = hp_clf.predict(val_hps)
    l4_val_preds = l4_clf.predict(val_l4)

    hp_test_preds = hp_clf.predict(test_hps)
    l4_test_preds = l4_clf.predict(test_l4)

    # evaluate predictions
    hp_val_acc = accuracy_score(val_labels, hp_val_preds)
    l4_val_acc = accuracy_score(val_labels, l4_val_preds)

    hp_test_acc = accuracy_score(test_labels, hp_test_preds)
    l4_test_acc = accuracy_score(test_labels, l4_test_preds)

    out_df = out_df.append(pd.DataFrame([[vp, val_files, train_files, hp_val_acc, l4_val_acc, hp_test_acc, l4_test_acc]], columns=out_df.columns), ignore_index=True)

    # plot confusion matrix
    # hp val
    cmat = confusion_matrix(val_labels, hp_val_preds, normalize="true", labels=LABELS)
    df_cmat = pd.DataFrame(cmat, index=LABELS, columns=LABELS)
    sb.heatmap(df_cmat, annot=True, cmap=sb.cubehelix_palette(n_colors=999, dark=0.3))
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("CMat HP val set, Test: {}".format(vp))
    plt.savefig(fname=save_path_plots + "cmat_hp_val_{}.png".format(vp))
    plt.show()

    # l4 val
    cmat = confusion_matrix(val_labels, l4_val_preds, normalize="true", labels=LABELS)
    df_cmat = pd.DataFrame(cmat, index=LABELS, columns=LABELS)
    sb.heatmap(df_cmat, annot=True, cmap=sb.cubehelix_palette(n_colors=999, dark=0.3))
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("CMat l4 val set, Test: {}".format(vp))
    plt.savefig(fname=save_path_plots + "cmat_l4_val_{}.png".format(vp))
    plt.show()

    # hp test
    cmat = confusion_matrix(test_labels, hp_test_preds, normalize="true", labels=LABELS)
    df_cmat = pd.DataFrame(cmat, index=LABELS, columns=LABELS)
    sb.heatmap(df_cmat, annot=True, cmap=sb.cubehelix_palette(n_colors=999, dark=0.3))
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("CMat hp test set, Test: {}".format(vp))
    plt.savefig(fname=save_path_plots + "cmat_hp_test_{}.png".format(vp))
    plt.show()

    # l4 test
    cmat = confusion_matrix(test_labels, l4_test_preds, normalize="true", labels=LABELS)
    df_cmat = pd.DataFrame(cmat, index=LABELS, columns=LABELS)
    sb.heatmap(df_cmat, annot=True, cmap=sb.cubehelix_palette(n_colors=999, dark=0.3))
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("CMat l4 test set, Test: {}".format(vp))
    plt.savefig(fname=save_path_plots + "cmat_l4_test_{}.png".format(vp))
    plt.show()

out_df.to_csv("../output/val_test_accs.csv", index=False)