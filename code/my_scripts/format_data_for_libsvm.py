import argparse
import pandas as pd
import os
from random import shuffle


def parse_args():

    parser = argparse.ArgumentParser(description="Takes the meta_data, l4, and l4_no_pca files for the train, val and test sets "
                                                 "of students and returns them in libsvc format.")
    parser.add_argument("--in_path", dest="in_path", help="Path to folder with meta_data, l4 and l4_no_pca", type=str,
                        required=True)
    parser.add_argument("--test", dest="test", help="Id of VP for test", type=int, required=True)
    parser.add_argument("--train", dest="train", help="String of train vp ids, comma separated: 1,12,15,9,2", type=str,
                        required=True)
    parser.add_argument("--val", dest="val", help="String of val vp ids, comma separated: 1,12,15,9,2", type=str,
                        required=True)
    parser.add_argument("--out_path", dest="out_path", help="Path for outputting the formatted files.", type=str,
                        required=True)

    return parser.parse_args()


def to_libsvm_format(df: pd.DataFrame, labels: pd.DataFrame):
    out = ""

    for idx, row in df.iterrows():
        out += "{} ".format(int(labels.iloc[idx].values[0]))

        row_list = row.values.tolist()
        for i in range(len(row_list)):
            out += "{}:{} ".format(i+1, row_list[i])
        out += "\n"

    return out


# args = parse_args()
# test = "{}.csv".format(args.test)
# vals = args.val.split(",")
# val = ["{}.csv".format(x) for x in vals]
# trains = args.train.split(",")
# train = ["{}.csv".format(x) for x in trains]

in_path = "../../source/train_val_test_sets/"        # args.in_path
out_path = "../../source/libsvm_train_test_val/"     # args.out_path
out_df = pd.DataFrame([], columns=["test", "val", "train"])

files = os.listdir(in_path + "meta_data/")

count = 1

for file in files:
    print("{} {}/{}".format(file, count, len(files)))
    count += 1
    test = file
    copy = files.copy()
    copy.remove(test)
    shuffle(copy)
    train = copy[:9]
    val = copy[9:]

    out_df = out_df.append(pd.DataFrame([[test[:-4], str([x[:-4] for x in val])[1:-1], str([x[:-4] for x in train])[1:-1]]],
                               columns=out_df.columns))

    test_label_df = pd.read_csv("{}labels/{}".format(in_path, test))
    test_hp_df = pd.read_csv("{}hp/{}".format(in_path, test))
    test_l4_df = pd.read_csv("{}l4/{}".format(in_path, test))
    test_l4_no_pca_df = pd.read_csv("{}l4_no_pca/{}".format(in_path, test))

    val_label_df = pd.DataFrame([])
    val_hp_df = pd.DataFrame([])
    val_l4_df = pd.DataFrame([])
    val_l4_no_pca_df = pd.DataFrame([])

    for vvp in val:
        val_label_df = val_label_df.append(pd.read_csv("{}labels/{}".format(in_path, vvp)))
        val_hp_df = val_hp_df.append(pd.read_csv("{}hp/{}".format(in_path, vvp)))
        val_l4_df = val_l4_df.append(pd.read_csv("{}l4/{}".format(in_path, vvp)))
        val_l4_no_pca_df = val_l4_no_pca_df.append(pd.read_csv("{}l4_no_pca/{}".format(in_path, vvp)))

    train_label_df = pd.DataFrame([])
    train_hp_df = pd.DataFrame([])
    train_l4_df = pd.DataFrame([])
    train_l4_no_pca_df = pd.DataFrame([])

    for tvp in train:
        train_label_df = train_label_df.append(pd.read_csv("{}labels/{}".format(in_path, tvp)))
        train_hp_df = train_hp_df.append(pd.read_csv("{}hp/{}".format(in_path, tvp)))
        train_l4_df = train_l4_df.append(pd.read_csv("{}l4/{}".format(in_path, tvp)))
        train_l4_no_pca_df = train_l4_no_pca_df.append(pd.read_csv("{}l4_no_pca/{}".format(in_path, tvp)))

    test_hp_df.to_csv("{}hp/{}_test.csv".format(out_path, test[:-4]), index=False, header=False)
    train_hp_df.to_csv("{}hp/{}_train.csv".format(out_path, test[:-4]), index=False, header=False)
    val_hp_df.to_csv("{}hp/{}_val.csv".format(out_path, test[:-4]), index=False, header=False)

    test_l4_df.to_csv("{}l4/{}_test.csv".format(out_path, test[:-4]), index=False, header=False)
    train_l4_df.to_csv("{}l4/{}_train.csv".format(out_path, test[:-4]), index=False, header=False)
    val_l4_df.to_csv("{}l4/{}_val.csv".format(out_path, test[:-4]), index=False, header=False)

    test_l4_no_pca_df.to_csv("{}l4_no_pca/{}_test.csv".format(out_path, test[:-4]), index=False, header=False)
    train_l4_no_pca_df.to_csv("{}l4_no_pca/{}_train.csv".format(out_path, test[:-4]), index=False, header=False)
    val_l4_no_pca_df.to_csv("{}l4_no_pca/{}_val.csv".format(out_path, test[:-4]), index=False, header=False)

out_df.to_csv("{}split_info.csv".format(out_path), index=False)

