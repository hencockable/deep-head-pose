#%%

# import


import os
import cv2
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image
from time import perf_counter

import datasets, hopenet, utils
import dlib
from math import floor, ceil

#%%

# define input parameters
start = perf_counter()
# if __name__ == '__main__':
cudnn.enabled = True

batch_size = 1
gpu = 0
snapshot_path = '../../../source/hopenet_robust_alpha1.pkl'
out_dir = '../output/video'
video_path = "../../../data/testvid_small.mp4"
face_model = "../../source/mmod_human_face_detector.dat"
output_string = "hopenet_dlib"
fps = 1
n_frames = 150

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if not os.path.exists(video_path):
    raise Warning("Video does not exist")

#%%

# initialsize models

# ResNet50 structure
model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

# Dlib face detection model
cnn_face_detector = dlib.cnn_face_detection_model_v1(face_model)

print('Loading snapshot.')
# Load snapshot
saved_state_dict = torch.load(snapshot_path)
model.load_state_dict(saved_state_dict)

print('Loading data.')

transformations = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224), transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

model.cuda(gpu)

print('Ready to test network.')

#%%

# Test the Model - preps
model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
total = 0

idx_tensor = [idx for idx in range(66)]
idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

video = cv2.VideoCapture(video_path)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out1 = cv2.VideoWriter('output/video/output-hopenet_dlib1.avi' , fourcc, 1, (width, height))
out30 = cv2.VideoWriter('output/video/output-hopenet_dlib30.avi' , fourcc, 30, (width, height))

txt_out = open('output/video/output-%s.txt' % output_string, 'w')

#%%

# Test the Model - iteration over frames
frame_num = 1

while frame_num <= n_frames:
    print("Frame", frame_num)

    ret, frame = video.read()
    if not ret:
        print("Needs to reload video")
        txt_out.write("Runtime: " + str(perf_counter() - start) + "seconds")
        out1.release()
        out30.release()
        video.release()
        txt_out.close()
        break

    cv2_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Dlib detect
    dets = cnn_face_detector(cv2_frame, 1)

    for idx, det in enumerate(dets):
        # Get x_min, y_min, x_max, y_max, conf
        x_min = det.rect.left()
        y_min = det.rect.top()
        x_max = det.rect.right()
        y_max = det.rect.bottom()
        conf = det.confidence

        if conf > 0.5: # original conf > 1.0 ?????
            bbox_width = abs(x_max - x_min)
            bbox_height = abs(y_max - y_min)
            x_min -= 2 * bbox_width / 4
            x_max += 2 * bbox_width / 4
            y_min -= 3 * bbox_height / 4
            y_max += bbox_height / 4
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(frame.shape[1], x_max)
            y_max = min(frame.shape[0], y_max)
            print(x_max, x_min, y_max, y_min)
            # Crop image
            img = cv2_frame[floor(y_min):ceil(y_max), floor(x_min):ceil(x_max)] # ceil and floor added by me because of errors due to float indices
            img = Image.fromarray(img)

            # Transform
            img = transformations(img)
            img_shape = img.size()
            img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
            img = Variable(img).cuda(gpu)

            # compute head pose
            yaw, pitch, roll = model(img)

            # adjust headpose
            yaw_predicted = F.softmax(yaw)
            pitch_predicted = F.softmax(pitch)
            roll_predicted = F.softmax(roll)

            # Get continuous predictions in degrees.
            yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
            roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

            # Print new frame with cube and axis
            txt_out.write(str(frame_num) + ' %f %f %f\n' % (yaw_predicted, pitch_predicted, roll_predicted))
            utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx=(x_min + x_max) / 2,
                                            tdy=(y_min + y_max) / 2, size=bbox_height / 2)

    out1.write(frame)
    out30.write(frame)
    frame_num += 1

txt_out.write("Runtime: " + str(perf_counter() - start) + "seconds")

out1.release()
out30.release()
video.release()
txt_out.close()

#%%


