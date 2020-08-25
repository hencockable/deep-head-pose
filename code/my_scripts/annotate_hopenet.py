import cv2
import sys
from random import randint
from math import cos, sin
import numpy as np


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):

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


def test_input_for_int(input):
    try:
        dummy = int(input)
    except ValueError:
        return -1
    return dummy

"""
Annotation script for one person in the classroom videos.

Labels:
    0: Teacher/whiteboard (keycode 119 - W)
    1: desk in front (keycode 115 - S)
    2: left neighbour (keycode 97 - A)
    3: right neighbour (keycode 100 - D)
    4: other (keycode 32 - Space)
"""

key_dict = {119: 0,
            115: 1,
            97: 2,
            100: 3,
            32: 4}

video_path = "../../mount/storage/local/wct/meta_data/videos/0925_G12_Chemistry_cut.mkv"

video = cv2.VideoCapture(video_path)
cv2.namedWindow("das", cv2.WINDOW_NORMAL)
meta = "../../mount/storage/local/wct/meta_data/videos_hendrik/0925_G12_Chemistry_cut_hopenet.txt"
frame_num = 0

with open(meta, 'r') as f:
    meta_line_list = f.read().splitlines()
    meta_line_list = meta_line_list[1:]  # remove header

txt_out = open("../../source/annotations_0925_G12_Chemistry_cut_VP2.txt", "w")
previous = ""

idx = 0
while idx < len(meta_line_list):
    line = meta_line_list[idx]
    line = line.strip('\n')
    line = line.split(" ")
    det_frame_num = int(line[0])

    # Save all frames as they are if they don't have bbox annotation.
    while frame_num < det_frame_num:
        ret, frame = video.read()
        if ret is False:
            video.release()
            txt_out.close()
            sys.exit(0)
        frame_num += 1

    ret, frame = video.read()
    if ret is False:
        break

    cv2.putText(frame, "z: Delete previous, j: Skip frame, p: Exit", (10, int(video.get(4)-50)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), thickness=5)

    frame_ori = frame.copy()
    frame_dict = {}

    while True:
        x_min, y_min, x_max, y_max = int(float(line[5])), int(float(line[6])), int(float(line[7])), int(float(line[8]))
        yaw, pitch, roll = float(line[1]), float(line[2]), float(line[3])
        bbox_in_frame = line[4]

        frame_dict[bbox_in_frame] = {"yaw": yaw,
                                     "pitch": pitch,
                                     "roll": roll,
                                     "x_min": x_min,
                                     "y_min": y_min,
                                     "x_max": x_max,
                                     "y_max": y_max}

        bbox_width = abs(x_max - x_min)
        bbox_height = abs(y_max - y_min)

        draw_axis(frame, yaw, pitch, roll, tdx=(x_min + x_max) / 2, tdy=(y_min + y_max) / 2, size=bbox_height / 2)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), thickness=5)
        cv2.putText(frame, bbox_in_frame, (x_min, y_min-20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), thickness=5)

        # Peek next frame detection
        try:
            next_frame_num = int(meta_line_list[idx + 1].strip('\n').split(' ')[0])
        except IndexError:
            break

        if next_frame_num == det_frame_num:
            idx += 1
            line = meta_line_list[idx].strip('\n').split(' ')
            det_frame_num = int(line[0])
        else:
            break

    frame = cv2.resize(frame, (1024, 576), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("das", frame)
    wkey_bbox = chr(cv2.waitKey(0))

    if wkey_bbox is "z":
        previous = ""
        cv2.putText(frame, "Previous removed", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), thickness=3)
        frame = cv2.resize(frame, (1024, 576), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("das", frame)
        wkey_bbox = chr(cv2.waitKey(0))

    elif wkey_bbox is "j":
        if previous:
            txt_out.write(previous)
            print(previous)
        previous = ""
        frame_num += 1
        idx += 1
        continue

    elif wkey_bbox is "p":
        break

    while test_input_for_int(wkey_bbox) not in [*range(len(frame_dict.keys()))]:
        cv2.putText(frame, "NOT A BBOX", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (randint(0, 255), randint(0, 255), randint(0, 255)), thickness=3)
        frame = cv2.resize(frame, (1024, 576), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("das", frame)
        wkey_bbox = chr(cv2.waitKey(0))

    data = frame_dict[wkey_bbox]
    cv2.rectangle(frame_ori, (data["x_min"], data["y_min"]), (data["x_max"], data["y_max"]), (0, 255, 0), thickness=5)
    draw_axis(frame_ori, data["yaw"], data["pitch"], data["roll"], tdx=(data["x_min"] + data["x_max"]) / 2, tdy=(data["y_min"] + data["y_max"]) / 2, size=abs(data["y_max"] - data["y_min"]) / 2)
    frame_ori = cv2.resize(frame_ori, (1024, 576), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("das", frame_ori)
    wkey_label = cv2.waitKey(0)

    while wkey_label not in key_dict.keys():
        cv2.putText(frame_ori, "NOT A LABEL (W=Teacher/Whiteboard, S=Desk, A=Left, D=Right, Space=Other", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (randint(0, 255), randint(0, 255), randint(0, 255)), thickness=1)
        frame_ori = cv2.resize(frame_ori, (1024, 576), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("das", frame_ori)
        wkey_label = cv2.waitKey(0)

    if previous:
        txt_out.write(previous)
        print(previous)

    previous = "{} {} {} {}\n".format(data["yaw"], data["pitch"], data["roll"], key_dict[wkey_label])

    frame_num += 1
    idx += 1

if previous:
    txt_out.write(previous)
    print(previous)

txt_out.close()


