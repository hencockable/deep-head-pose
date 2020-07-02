import argparse, os, sys
import pandas as pd


# Argument parser
parser = argparse.ArgumentParser(description='Pre-processing of bbox annotations to fit Hopenet.')
parser.add_argument('--file_path', dest='file_path', help='Path to bbox file', type=str, required=True)
parser.add_argument("--out_dir", dest="out_dir", help="Output directory", type=str, required=True)
args = parser.parse_args()

# get args
path = args.file_path
outdir = args.out_dir

if not os.path.exists(outdir):
    os.makedirs(outdir)

if not os.path.exists(path):
    sys.exit('File does not exist')

# get file name
name = os.path.splitext(os.path.basename(path))[0]
out_name = outdir + name + "_hopenet.csv"

# read bbox .csv file
df = pd.read_csv(path)

# select for Hopenet required columns (n_frame x_min y_min x_max y_max confidence)
out = df[["frame_num", "bb_x1", "bb_y1", "bb_x2", "bb_y2", "score"]].copy()

# save as new .csv file
out.to_csv(out_name, index=False)
print("Saved", out_name)





