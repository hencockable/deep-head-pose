import pandas as pd
import os

# all vps
# 'VP3', 'VP8', 'VP7', 'VP14', 'VP2', 'VP13', 'VP16', 'VP12', 'VP10', 'VP19', 'VP18', 'VP5', 'VP11'

data_path = "../../source/train_val_test_vps/data/"
#l4_path = "../../source/train_val_test_vps/l4/"
l4_no_pca_path = "../../source/train_val_test_vps/l4_no_pca/"
out_path = "../../source/train_val_test_sets/"

files = set(os.listdir(data_path))
vps = set([file.split("_")[4] for file in files])

for vp in vps:
    #data_df = pd.DataFrame([])
    #l4_df = pd.DataFrame([])
    l4_no_pca_df = pd.DataFrame([])
    for file in files:
        if vp in file:
            #data_df = data_df.append(pd.read_csv(data_path + file))
            #l4_df = l4_df.append(pd.read_csv(l4_path + file[:-8] + "l4.csv"))
            l4_no_pca_df = l4_no_pca_df.append(pd.read_csv(l4_no_pca_path + file[:-8] + "l4_no_pca.csv"))


    #data_df.to_csv("{}data/{}.csv".format(out_path, vp), index=False)
    #l4_df.to_csv("{}l4/{}.csv".format(out_path, vp), index=False)
    l4_no_pca_df.to_csv("{}l4_no_pca/{}.csv".format(out_path, vp), index=False)



