import numpy as np
import os
import pandas as pd
import math
import cv2

## Tomorrow add G0 in empty places
## need take into acount number of frames in the end of the file

target_folder = os.path.join("transcriptions_activities")
source_folder = os.path.join("original_labeles_fixed")
video_folder = os.path.join("videos")
list_of_files = os.listdir(source_folder)
dict={
"no gesture" : "G0",
"One Shot needle passing" : "G1",
"Out to in needle passing": "G1",
"In to out needle passing" :"G1",
"pull the suture": "G2",
"Instrument tie- loop suture twice around the needle driver": "G3","Instrumental tie- loop wire once around the needle driver": "G3","Lay the knot" : "G4",
"Cut the suture" :"G5"}


for file_name in list_of_files:
    text_file =[]
    text_line = []

    csv_read = pd.read_csv(os.path.join(source_folder,file_name))
    data = csv_read.values[15:,[0,5,8]]
    for i,line in enumerate(data):
        if line[2] == "START":
            G = (dict[line[1]])
            frame_num_start = math.floor(float(line[0]) * 30)
            if i == 0 and frame_num_start != 0:
                text_line.append(0)
                text_line.append(frame_num_start-1)
                text_line.append("G0")
                text_file.append(text_line)
                text_line = []

            for j in range(i+1, len(data)):

                if data[j,2] == "STOP" and dict[data[j,1]] == G:
                    frame_num_stop = math.floor(30*float(data[j,0]))-1
                    if frame_num_start != text_file[-1][1]+1:
                        text_line.append(text_file[-1][1]+1)
                        text_line.append(frame_num_start - 1)
                        text_line.append("G0")
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
    if total_number_of_frames > text_file[-1][1]:
        text_line.append(text_file[-1][1] + 1)
        text_line.append(total_number_of_frames)
        text_line.append("G0")
        text_file.append(text_line)
        text_line = []

    df = pd.DataFrame(text_file)
    df.to_csv(os.path.join(target_folder,file_name[:-4] +".txt"), index=False,header=False,sep=" ")
