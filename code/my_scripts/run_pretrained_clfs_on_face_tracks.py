import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib.colors import LinearSegmentedColormap


def collapse_list(lst):
    out = []

    current_label = lst[0]
    current_len = 1
    for j in range(len(lst)):
        try:
            if lst[j] == lst[j + 1]:
                current_len += 1
            else:
                out.append((current_label, current_len))
                current_label = lst[j + 1]
                current_len = 1
        except IndexError:
            out.append((current_label, current_len))

    return out


vps = ["3", "8", "11", "12", "14", "18"]

file = "0927_G08_French_cut"
clfs_path = "../output/clfs/annotations_{}_VP{}_1sec_skip9secs.csv_joined_l2l3/"
face_track_path = "/home/hendrik/PycharmProjects/Master/Hopenet/source/face_tracks/face_track_{}_VP{}_filled.csv"
face_detection_df = pd.read_csv("/home/hendrik/PycharmProjects/Master/Hopenet/source/face_detection/{}.csv".format(file))
total_frames = face_detection_df.frame_num.max() + 1

preds_17 = []
preds_all = []
rows_17 = []
rows_all = []

for vp in vps:
    clf_17 = joblib.load(clfs_path.format(file, vp) + "SVC_17.sav")
    clf_all = joblib.load(clfs_path.format(file, vp) + "SVC_all.sav")

    face_track_df = pd.read_csv(face_track_path.format(file, vp))

    pred_17 = list(clf_17.predict(face_track_df[["yaw", "pitch", "roll"]]))
    pred_all = list(clf_all.predict(face_track_df[["yaw", "pitch", "roll"]]))

    pred_df = pd.DataFrame(list(zip(face_track_df.frame_num, pred_all, pred_17)), columns=["frame_num", "SVCall", "SVC17"])

    filled_17 = []
    filled_all =[]

    for i in range(total_frames):
        if i in pred_df.frame_num.values:
            row = pred_df[pred_df.frame_num == i]
            filled_17.append(int(row.SVC17))
            filled_all.append(int(row.SVCall))
        else:
            filled_17.append(-1)
            filled_all.append(-1)

    preds_all.append(filled_all)
    preds_17.append(filled_17)

    rows_all.append("VP{}".format(vp))
    rows_17.append("VP{}".format(vp))


fig, ax = plt.subplots()
sb.heatmap(preds_all, ax=ax)
ax.set_yticklabels(rows_all)
ax.set_xticklabels([])
plt.xlabel("Video duration [frames]")
plt.title("SVC - all samples")
plt.show()

fig, ax = plt.subplots()
sb.heatmap(preds_17, ax=ax)
ax.set_yticklabels(rows_17)
ax.set_xticklabels([])
plt.xlabel("Video duration [frames]")
plt.title("SVC - 17 samples per label")
plt.show()

    # x = collapse_list(filled_17)
    # y = collapse_list(filled_all)
    #
    #
    # print(len(x))
    # print(len(y))


