import argparse, os
import pandas as pd


# Argument parser
parser = argparse.ArgumentParser(description='Pre-processing of bbox annotations to fit Hopenet.')
parser.add_argument('--file_path', dest='file_path', help='Path to bbox file', type=str, required=True)
args = parser.parse_args()

# get path to bbox .csv file
path = args.file_path
name = os.path.splitext(path)[0]

# read bbox .csv file
df = pd.read_csv(path)

# select for Hopenet required columns (n_frame x_min y_min x_max y_max confidence)
out = df[["frame_num", "bb_x1", "bb_y1", "bb_x2", "bb_y2", "score"]].copy()

# save as new .csv file
out.to_csv("{}_hopenet.csv".format(name), index=False)





