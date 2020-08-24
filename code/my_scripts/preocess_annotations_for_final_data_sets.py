import pandas as pd
import os

annotations_path = "../../source/annotations/annotations_{}/perSecond/"
out_path = "../../source/train_val_test_vps/"

videos = ["0927_G08_French_cut", "0930_G08_IMP_cut"]

for video in videos:
    all_data = pd.read_csv("/storage/local/wct/data/videos_hendrik/{}_l4_no_pca.csv".format(video))

    annotations = os.listdir(annotations_path.format(video))

    for annotation in annotations:
        # cols = ["file", "vp", "frame_num", "face_id", "total_detected_faces", "x_min", "y_min", "x_max", "y_max", "x1", "y1", "x2", "y2", "x3",
        #         "y3", "x4", "y4", "x5", "y5", "yaw", "pitch", "roll", "label"]
        # out_df = pd.DataFrame([], columns=cols)
        l4_df = pd.DataFrame([])

        vp = annotation.split("_")[5][2:]
        annotation_df = pd.read_csv(annotations_path.format(video) + annotation)

        for _, row in annotation_df.iterrows():
            if row.frame_num in all_data.frame_num.values and row.face_id in all_data.face_id.values:
                sample = all_data.loc[(all_data.frame_num == row.frame_num) & (all_data.face_id == row.face_id)]
                sample = sample.drop(columns=["frame_num", "face_id"]).values.tolist()[0]

                # l4 = sample.l4.values[0][1:-1].replace("\n", "").replace("  ", " ").split(" ")
                # try:
                #     l4 = list(map(float, l4))
                # except ValueError:
                #     l4 = [x for x in l4 if x != ""]
                #     l4 = list(map(float, l4))

                # l4_df = l4_df.append(pd.DataFrame([l4]), ignore_index=True)
                l4_df = l4_df.append(pd.DataFrame([sample]), ignore_index=True)

                #sample = sample.drop(columns=["l4", "score"]).values.tolist()[0]
                #data = [video, vp]
                #data.extend(sample)
                #data.append(row.label)
                #out_df = out_df.append(pd.DataFrame([data], columns=out_df.columns), ignore_index=True)

        #out_df.to_csv("{}data/{}_VP{}_data.csv".format(out_path, video, vp), index=False)
        l4_df.to_csv("{}l4_no_pca/{}_VP{}_l4_no_pca.csv".format(out_path, video, vp), index=False)
        #print(out_df.shape)
        print(l4_df.shape)
