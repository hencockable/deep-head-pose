import cv2
import pandas as pd
import numpy as np
from math import cos, sin


def make_video(blank_img, v_out, width, height, start, stop, angles):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(v_out, fourcc, 24, (width, height))

    for frame_id in range(start, stop):
        sub_df = df[df.frame == frame_id]
        copy = blank_img.copy()

        for index, row in sub_df.iterrows():
            x_min, y_min, x_max, y_max = int(row.x_min), int(row.y_min), int(row.x_max), int(row.y_max)

            cv2.rectangle(copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)

            if angles:
                bbox_height = abs(y_max - y_min)
                draw_axis(copy, row.yaw, row.pitch, row.roll, tdx=(x_min + x_max) / 2, tdy=(y_min + y_max) / 2,
                          size=int(bbox_height / 2))

        out.write(copy)

    out.release()


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img


def make_single_frame(blank_img, yaw, pitch, roll, img_out):
    copy = blank_img.copy()

    draw_axis(copy, yaw, pitch, roll, size=200)

    cv2.imwrite(img_out, copy)


# for creating videos with blank bg but face bbs and/or HP angles drawn on it
#df = pd.read_csv("../../source/0925_G12_Chemistry_cut_hopenet.txt", delimiter=" ")
video_out = "Hopenet/code/output/video/white_example_hp_angles.avi"
height = 1728
width = 3072
start = 0
stop = 120

# for creating singles frames as examples for head poses
img_out = "Hopenet/code/output/frames/white_example_hp_angles_label3.png"
yaw = 84.109604
pitch = -24.477439999999998
roll = -24.480713

# create white image of original video resolution
blank_image = np.zeros((height, width, 3), np.uint8)
blank_image[:] = (255, 255, 255)

make_single_frame(blank_image, yaw, pitch, roll, img_out)







