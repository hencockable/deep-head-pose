
# Imports
import sys, os, argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image
import pandas as pd

import my_hopenet, utils
from sklearn.decomposition import PCA


# Argument parser
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='', type=str)
    parser.add_argument('--video', dest='video_path', help='Path of video')
    parser.add_argument('--bboxes', dest='bboxes', help='Bounding box annotations of frames')
    parser.add_argument('--output_string', dest='output_string', help='String appended to output file')
    parser.add_argument('--fps', dest='fps', help='Frames per second of source video', type=float, default=30.)
    parser.add_argument(("--out_dir"), dest="outdir", help="Output directory", type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True

    batch_size = 1
    gpu = args.gpu_id
    snapshot_path = args.snapshot
    out_dir = args.outdir
    video_path = args.video_path

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(args.video_path):
        sys.exit('Video does not exist')

    # ResNet50 structure
    model = my_hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    print('Loading snapshot.')
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)

    print('Loading data.')

    transformations = transforms.Compose([transforms.Scale(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    model.cuda(gpu)

    print('Ready to test network.')

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    total = 0

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    video = cv2.VideoCapture(video_path)

    # New cv2
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(out_dir + '%s_hopenet.avi' % args.output_string, fourcc, args.fps, (width, height))

    # txt_out = open(out_dir + '%s_head_poses.csv' % args.output_string, 'w')
    # txt_out.write("frame_num,face_id,total_detected_faces,x_min,y_min,x_max,y_max,score,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,"
    #               "yaw,pitch,roll,l4\n")
    # l4_out = open("{}{}_l4_data.csv".format(out_dir, args.output_string), "w")
    # l4_out.write("l4")

    out_df = pd.DataFrame([], columns=["frame_num","face_id","total_detected_faces","x_min","y_min","x_max","y_max",
                                       "score","x1","y1","x2","y2","x3","y3","x4","y4","x5","y5","yaw","pitch","roll"])
    l4s = []

    frame_num = 0   # hendrik

    # with open(args.bboxes, 'r') as f:
    #     bbox_line_list = f.read().splitlines()
    #     bbox_line_list = bbox_line_list[1:]     # remove header

    bbox_line_df = pd.read_csv(args.bboxes)

    idx = 0
    while idx < bbox_line_df.shape[0]:
        line = bbox_line_df.iloc[idx]
        det_frame_num = int(line.frame_num)

        print("{} \\ {} - {} -> {:.2f}".format(idx, bbox_line_df.shape[0], args.output_string, idx/bbox_line_df.shape[0]))

        # Save all frames as they are if they don't have bbox annotation.
        while frame_num < det_frame_num:
            ret, frame = video.read()
            if ret == False:
                out.release()
                video.release()
                #txt_out.close()
                sys.exit(0)
            out.write(frame)
            frame_num += 1

        # Start processing frame with bounding box
        ret,frame = video.read()
        if ret == False:
            print("Couldnt read next frame.")
            break
        cv2_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        while True:
            x_min, y_min, x_max, y_max = int(line.bb_x1), int(line.bb_y1), int(line.bb_x2), int(line.bb_y2)

            bbox_width = abs(x_max - x_min)
            bbox_height = abs(y_max - y_min)
            # x_min -= 3 * bbox_width / 4
            # x_max += 3 * bbox_width / 4
            # y_min -= 3 * bbox_height / 4
            # y_max += bbox_height / 4
            x_min -= 50
            x_max += 50
            y_min -= 50
            y_max += 30
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(frame.shape[1], x_max)
            y_max = min(frame.shape[0], y_max)
            # Crop face loosely
            img = cv2_frame[y_min:y_max, x_min:x_max]
            img = Image.fromarray(img)

            # Transform
            img = transformations(img)
            img_shape = img.size()
            img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
            img = Variable(img).cuda(gpu)

            yaw, pitch, roll, l4 = model(img)

            yaw_predicted = F.softmax(yaw)
            pitch_predicted = F.softmax(pitch)
            roll_predicted = F.softmax(roll)

            # Get continuous predictions in degrees.
            yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
            roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

            # Print new frame with cube and axis
            # txt_out.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
            #     frame_num, line.face_id, line.total_detected_faces, x_min, y_min, x_max, y_max, line.score,
            #     line.x1, line.y1, line.x2, line.y2, line.x3, line.y3, line.x4, line.y4, line.x5, line.y5, yaw_predicted,
            #     pitch_predicted, roll_predicted
            # ))
            # l4_out.write("{}\n".format(str(l4)[1:-1]))

            l4s.append(l4)

            data = [frame_num, line.face_id, line.total_detected_faces, line.bb_x1, line.bb_y1, line.bb_x2, line.bb_y2,
                    line.score, line.x1, line.y1, line.x2, line.y2, line.x3, line.y3, line.x4, line.y4, line.x5, line.y5,
                    float(yaw_predicted), float(pitch_predicted), float(roll_predicted)]
            data_df = pd.DataFrame([data], columns=out_df.columns)
            out_df = out_df.append(data_df, ignore_index=True)

            # txt_out.write(str(frame_num) + ' %f %f %f %s %s %s %s %s\n' % (yaw_predicted, pitch_predicted, roll_predicted, bbox_in_frame, x_min, y_min, x_max, y_max))
            # utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, (x_min + x_max) / 2, (y_min + y_max) / 2, size = bbox_width)
            utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/2)
            # Plot expanded bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)

            # Peek next frame detection
            try:
                next_frame_num = int(bbox_line_df.iloc[idx+1].frame_num)
            except Exception:
                print("Reached end of file.")
                next_frame_num = -1
            # print 'next_frame_num ', next_frame_num
            if next_frame_num == det_frame_num:
                idx += 1
                line = bbox_line_df.iloc[idx]
                det_frame_num = int(line.frame_num)
            else:
                break

        idx += 1
        out.write(frame)
        frame_num += 1

    out.release()
    video.release()
    #txt_out.close()
    #l4_out.close()

    print("Starting PCA reduction to 100 features.")
    pca = PCA(n_components=100)
    preds = pca.fit_transform(l4s)
    print("PCA finished.")

    out_df["l4"] = list(preds)

    out_df.to_csv("{}{}_l4.csv".format(out_dir, args.output_string), index=False)
    print("Written to file - Exiting.")