import numpy as np
import os
import pandas as pd
import math
import cv2
def marge_identicals(text_file):
    new_text_file = []
    for i, line in enumerate(text_file):
        if i == 0:
            new_text_file.append(line)
        elif text_file[i][-1] == text_file[i - 1][-1]:
            new_text_file[-1][1] = line[1]
        else:
            new_text_file.append(line)
    return new_text_file

Mode = "Right"
source_folder = os.path.join("oreginal_tool_labels")
video_folder = os.path.join("videos")
list_of_files = os.listdir(source_folder)

dict_LEFT={
"free_hand_L" : "T0", "working_hand_L" : "T0", "needle_drive_L": "T1", "forceps_L": "T2", "scissors_L":"T3"
}

dict_Right={
"free_hand_R" : "T0", "working_hand_R" : "T0", "needle_drive_R": "T1", "forceps_R": "T2", "scissors_R":"T3"
}

if Mode == "Right":
    dict = dict_LEFT
    target_folder = os.path.join("transcriptions_tools_right_new")

else:
    dict =dict_Right
    target_folder = os.path.join("transcriptions_tools_left_new")

for file_name in list_of_files:
    first_time = True
    text_file =[]
    text_line = []

    csv_read = pd.read_csv(os.path.join(source_folder,file_name))
    data = csv_read.values[15:,[0,5,8]]
    for i,line in enumerate(data):
        if line[1] not in dict:
            continue

        if line[2] == "START":
            G = (dict[line[1]])
            frame_num_start = math.floor(float(line[0]) * 30)
            if first_time:
                first_time = False
                frame_num_start = 0
            if i == 0 and frame_num_start != 0:
                text_line.append(0)
                text_line.append(frame_num_start-1)
                text_line.append("T0")
                text_file.append(text_line)
                text_line = []

            for j in range(i+1, len(data)):

                if data[j,1] in dict and data[j,2] == "STOP" and dict[data[j,1]] == G:
                    frame_num_stop = math.floor(30*float(data[j,0]))-1
                    if len(text_file)>0 and frame_num_start != text_file[-1][1]+1:
                        text_line.append(text_file[-1][1]+1)
                        text_line.append(frame_num_start - 1)
                        text_line.append("T0")
                        text_file.append(text_line)
                        text_line = []

                    text_line.append(frame_num_start)
                    text_line.append(frame_num_stop)
                    text_line.append(G)
                    text_file.append(text_line)
                    text_line = []
                    break

    video_full_name = file_name[:-4]
    if video_full_name[:4] in ["P016","P017","P018","P019","P020","P021"]:
        video_full_name = video_full_name +".wmv"
    else:
        video_full_name = video_full_name +".wm1"

    cap = cv2.VideoCapture(os.path.join(video_folder, video_full_name))
    total_number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    text_file[-1][1] = total_number_of_frames
    text_file = marge_identicals(text_file)

    df = pd.DataFrame(text_file)
    df.to_csv(os.path.join(target_folder,file_name[:-4] +".txt"), index=False,header=False,sep=" ")
