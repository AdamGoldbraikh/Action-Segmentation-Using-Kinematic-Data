import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import (YEARLY, DateFormatter,rrulewrapper, RRuleLocator, drange)
import numpy as np
import datetime
import csv
import os
from scipy import signal
import copy

import numpy as np; np.random.seed(0)
import seaborn as sns
import matplotlib.pylab as plt



def pars_ground_truth(file_ptr):
    gt_source = file_ptr.read().split('\n')[:-1]
    contant = []
    for line in gt_source:
        info = line.split()
        contant.append(info)
    return contant


def get_row_list(row: list, col_dict: dict):
    """
    get the values from row that are in the columns in col_dict

    :param row: the row of values
    :param col_dict: a dictionary of the columns to take
    :return: a list of the values to be taken
    """
    row_list = []
    for idx, item in enumerate(row):
        if idx in col_dict.keys():
            item = float(item)
            row_list.append(item)
    return row_list


def create_header(row):
    """
    Taking all sensors, the X, Y, Z positions,  angles and rotation matrices
    :param row: The file header row
    :return: A list of all the columns to take and their indices
    """
    columns_list = []
    columns_dict = {}
    for idx, col in enumerate(row):
        if "Position" in col:
            col_name = col.split("/")[0].strip() + " " + col.split("/")[2].strip()
            columns_dict[idx] = col_name
            columns_list.append(col_name)
        if "Euler" in col:
            col_name = col.split("/")[0].strip() + " " + col.split("/")[1].strip() + " " + col.split("/")[2].strip()
            columns_dict[idx] = col_name
            columns_list.append(col_name)
        if "Rotation Matrix" in col:
            col_name = col.split("/")[0].strip() + " " + col.split("/")[1].strip() + " " + col.split("/")[2].strip()
            columns_dict[idx] = col_name
            columns_list.append(col_name)
    return columns_list, columns_dict


def fir_filter(vec, taps):
    yf=signal.lfilter(taps,1.0, x=vec-vec[0],axis=0)
    return yf+vec[0]


def create_clean_table(file_name,taps,with_filtration=False):

    is_relevant = False
    data = []
    with open(file_name, "r") as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if row != [] and row[0] == "Frame #":
                is_relevant = True
                col_list, col_dict = create_header(row)
                labels = col_list
                continue
            if not is_relevant:
                continue
            cur_out = get_row_list(row, col_dict)
            data.append(cur_out)
    df = pd.DataFrame(data,
                       columns=labels)
    if with_filtration:
        filtered_data = fir_filter(df.values, taps)
        df2 =pd.DataFrame(filtered_data,columns=labels)
    else:
        df2 =df
    alpha = 5.989833333
    i=0
    samples =[0]
    while i < df2.shape[0]-alpha:
        i = i+alpha
        samples.append(round(i))
    samples[-1]=samples[-1]-1
    return df2.iloc[samples]


def time_alignment_wrt_video(clean_df,delay):
    if delay == 0:
        return clean_df
    elif delay > 0:
        return clean_df.iloc[delay:]
    else:
        padding = np.zeros((abs(delay),clean_df.shape[1]))
        padding_df = pd.DataFrame(padding, columns=clean_df.columns)
        return padding_df.append(clean_df)


def create_clean_folder(delay_csv_path,taps,filtration=True):
    csv = pd.read_csv(delay_csv_path)
    for index, row in csv.iterrows():
        experiment=row[0]
        delay = row[1]
        clean_df = create_clean_table(os.path.join("Sensors_Raw_data",experiment+".exp"),taps,filtration)
        clean_df = time_alignment_wrt_video(clean_df,delay)
        if filtration:
            clean_df.to_csv(os.path.join("sensor_clean_with_filtration",experiment+".csv"),index=False, float_format='%.6f')
        else:
            clean_df.to_csv(os.path.join("sensor_clean_without_filtration", experiment + ".csv"), index=False, float_format='%.6f')


def save_as_numpy_files(source_path,target_path,sensor_disances=False,velocity= False):
    relevant_coordinates = np.array(
        list(range(0, 6)) + list(range(15, 21)) + list(range(30, 36)) + list(range(45, 51)) + list(
            range(60, 66)) + list(range(75, 81)))
    position_coordinates = np.array([0,1,2,6,7,8,12,13,14,18,19,20,24,25,26,30,31,32])
    srnsors_list=[np.array([0,1,2]),np.array([6,7,8]),np.array([12,13,14]), np.array([18,19,20]),np.array([24,25,26]),np.array([30,31,32])]


    for file in os.listdir(source_path):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            df=pd.read_csv(os.path.join(source_path,filename))
            features = df.values.T
            features = features[relevant_coordinates]
            #positional velocities only
            # velocity_a = features[position_coordinates][:,1:]
            # velocity_b = features[position_coordinates][:,0:-1]
            #            velocity = np.zeros_like(features[position_coordinates])
            if velocity:
                velocity_a = features[:,1:]
                velocity_b = features[:,0:-1]
                velocity = np.zeros_like(features)
                velocity[:,1:] = velocity_a - velocity_b

                features = velocity




            if sensor_disances:
                sensors_list =[0,1,2,3,4,5]
                sensor_pairs = [(a, b) for idx, a in enumerate(sensors_list) for b in sensors_list[idx + 1:]]
                distances=[]
                for pair in sensor_pairs:
                    sensor_a = srnsors_list[pair[0]]
                    sensor_b =srnsors_list[pair[1]]
                    dist_ = np.linalg.norm(features[sensor_a] - features[sensor_b], axis=0)
                    # dist_ = np.expand_dims(dist_, axis=1).T
                    distances.append(dist_)

                dist=np.stack(distances, axis=0)
                dist_velocity_a = dist[:,1:]
                dist_velocity_b = dist[:,0:-1]
                dist_velocity = np.zeros([dist.shape[0],features.shape[1]])
                dist_velocity[:,1:] = dist_velocity_a - dist_velocity_b

                features = np.concatenate((features, dist_velocity), axis=0)

            # #add speed
            # for sensor in sensors_list:
            #     sensor_position = srnsors_list[sensor]
            #     dist = np.linalg.norm(features[sensor_position][:,1:] - features[sensor_position][:,0:-1], axis=0)
            #     dist = np.concatenate((np.array([0.]), dist))
            #     dist= np.expand_dims(dist, axis=1).T
            #     features = np.concatenate((features, dist), axis=0)



            new_name= filename[:-4] +".npy"
            full_path = os.path.join(target_path,new_name)
            np.save(full_path, features)

            continue
        else:
            continue


def folds_list_of_files(fold_number,directory_path='./folds'):
    list_of_train_examples =[]
    for file in os.listdir(directory_path):
        filename = os.fsdecode(file)
        if filename.endswith(".txt") and "fold" in filename:
            if str(fold_number) in filename:
                continue
            else:
                file_ptr = open(os.path.join(directory_path, filename), 'r')
                list_of_train_examples = list_of_train_examples + file_ptr.read().split('\n')[:-1]
                file_ptr.close()
            continue
        else:
            continue
    return list_of_train_examples


def calculate_normalization_parameters(fold,with_filtration=True):
    features_list=[]
    list_of_train_examples = folds_list_of_files(fold)
    for csv in list_of_train_examples:
        file_neme = csv[:-4]+".npy"
        if with_filtration:
            numpy_files_path =os.path.join("kinematics_with_filtration_npy")
        else:
            numpy_files_path =os.path.join("kinematics_without_filtration_npy")
        full_file_path = os.path.join(numpy_files_path,file_neme)
        features = np.load(full_file_path)
        features_list.append(features)
    margged_feature_list =np.concatenate(features_list,axis=1)
    maximums = np.max(margged_feature_list,axis=1)
    minimums = np.min(margged_feature_list,axis=1)
    means = np.mean(margged_feature_list, axis=1)
    stds = np.std(margged_feature_list,axis=1)

    return [maximums, minimums, means,stds]


def calculate_normalization_parameters_all_folds():
    for i in range(5):
        params = calculate_normalization_parameters(i, with_filtration=True)
        df=pd.DataFrame(params,index=["max","min","mean","std"])
        df.to_csv(os.path.join("folds","std_params_fold_"+str(i))+".csv",index=True)


def dataset_analyzer_gesture(gt_path):
    action_dict = {"G0": 0, "G1": 1, "G2": 2, "G3": 3, "G4": 4, "G5": 5}

    num_of_samples_list = [0]*len(action_dict)
    duration_per_activity = [ [],[],[],[],[],[] ]
    num_of_samples_lists = []
    for file in os.listdir(gt_path):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            this_num_of_samples_list = num_of_samples_list.copy()
            file_ptr = open(os.path.join(gt_path,filename), 'r')
            gt_dada = pars_ground_truth(file_ptr)
            for line in gt_dada:
                this_num_of_samples_list[action_dict[line[2]]] += 1
                duration_per_activity[action_dict[line[2]]].append(int(line[1])-int(line[0]) + 1)
            num_of_samples_lists.append(this_num_of_samples_list)

    num_of_samples_lists = np.array(num_of_samples_lists)
    num_of_samples_lists_mean = num_of_samples_lists.mean(axis=0)
    num_of_samples_lists_sd = num_of_samples_lists.std(axis=0)
    num_of_samples_statistics = np.concatenate((np.expand_dims(num_of_samples_lists_mean,axis=1),np.expand_dims(num_of_samples_lists_sd,axis=1)),axis=1)
    duration_statistics=[]
    for action in duration_per_activity:
        duration_statistics.append([float(np.array(action).mean()),float(np.array(action).std())])
    duration_statistics = np.array(duration_statistics)/30
    statistics = np.concatenate((num_of_samples_statistics,duration_statistics),axis=1)
    headers =  ["number os samples mean", "number of samples sd", "activity duration mean [sec]", "activity duration sd [sec]"]
    activities = list(action_dict)
    df=pd.DataFrame(statistics,columns=headers, index=activities)
    df.to_csv("gesture_statistics.csv")

def dataset_analyzer_one_tools(gt_path):
    action_dict = {"T0": 0, "T1": 1, "T2": 2, "T3": 3}

    num_of_samples_list = [0]*len(action_dict)
    duration_per_activity = [ [],[],[],[]]
    num_of_samples_lists = []
    for file in os.listdir(gt_path):
        filename = os.fsdecode(file)
        if filename.endswith(".txt") and filename[2:4] != "31":
            this_num_of_samples_list = num_of_samples_list.copy()
            file_ptr = open(os.path.join(gt_path,filename), 'r')
            gt_dada = pars_ground_truth(file_ptr)
            for line in gt_dada:
                this_num_of_samples_list[action_dict[line[2]]] += 1
                duration_per_activity[action_dict[line[2]]].append(int(line[1])-int(line[0]) + 1)
            num_of_samples_lists.append(this_num_of_samples_list)

    return num_of_samples_lists, duration_per_activity

def dataset_analyzer_two_hands_tools(gt_path_left,gt_path_right):
    num_of_samples_lists_left, duration_per_activity_left = dataset_analyzer_one_tools(gt_path_left)
    num_of_samples_lists_right, duration_per_activity_right = dataset_analyzer_one_tools(gt_path_right)
    num_of_samples_lists = []
    for sample_left, sample_right in zip(num_of_samples_lists_left,num_of_samples_lists_right):
        sample = []
        sample.append(sample_left[0])
        sample.append(sample_right[0])
        sample.append(sample_left[1] + sample_right[1])
        sample.append(sample_left[2] + sample_right[2])
        sample.append(sample_left[3] + sample_right[3])
        num_of_samples_lists.append(sample)

    duration_per_activity = [ [],[],[],[],[0]]
    duration_per_activity[0] = duration_per_activity_left[0]
    duration_per_activity[1] = duration_per_activity_right[0]
    duration_per_activity[2] = duration_per_activity_left[1] + duration_per_activity_right[1]
    duration_per_activity[3] = duration_per_activity_left[2] + duration_per_activity_right[2]
    duration_per_activity[4] = duration_per_activity_left[3] + duration_per_activity_right[3]

    num_of_samples_lists = np.array(num_of_samples_lists)
    num_of_samples_lists_mean = num_of_samples_lists.mean(axis=0)
    num_of_samples_lists_sd = num_of_samples_lists.std(axis=0)
    num_of_samples_statistics = np.concatenate((np.expand_dims(num_of_samples_lists_mean,axis=1),np.expand_dims(num_of_samples_lists_sd,axis=1)),axis=1)
    duration_statistics=[]
    for action in duration_per_activity:
        duration_statistics.append([float(np.array(action).mean()),float(np.array(action).std())])
    duration_statistics = np.array(duration_statistics)/30
    statistics = np.concatenate((num_of_samples_statistics,duration_statistics),axis=1)

    headers =  ["number os samples mean", "number of samples sd", "activity duration mean [sec]", "activity duration sd [sec]"]
    activities =["LT0","RT0","T1","T2","T3"]
    df=pd.DataFrame(statistics,columns=headers, index=activities)
    df.to_csv("tools_statistics.csv")

def joint_distribution_analysis(gt_path_left,gt_path_right,gt_path_gesture,conditional_on_gesture):
    action_dict_gesture = {"G0": 0, "G1": 1, "G2": 2, "G3": 3, "G4": 4, "G5": 5}
    action_dict_tools = {"T0": 0, "T1": 1, "T2": 2, "T3": 3}
    action_dict_tools_2_hands = {"LT0": 0,"RT0": 1, "T1": 2, "T2": 3, "T3": 4}
    distribution= np.zeros((len(action_dict_gesture),len(action_dict_tools_2_hands)))
    for file in os.listdir(gt_path_gesture):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            file_ptr = open(os.path.join(gt_path_gesture,filename), 'r')
            gt_dada_gesture = pars_ground_truth(file_ptr)
            file_ptr = open(os.path.join(gt_path_left,filename), 'r')
            gt_dada_left = pars_ground_truth(file_ptr)
            file_ptr = open(os.path.join(gt_path_right,filename), 'r')
            gt_dada_right = pars_ground_truth(file_ptr)
            # last_index = min(gt_dada_gesture[-1][1],gt_dada_left[-1][1],gt_dada_right[-1][1])
            for gesture in gt_dada_gesture:
                G = gesture[2]
                init_gesture = int(gesture[0])
                end_gesture = int(gesture[1])
                for L_tool in gt_dada_left:
                    T = L_tool[2]
                    if T == "T0":
                        T = "LT0"
                    init_L = int(L_tool[0])
                    end_L =  int(L_tool[1])
                    intersection = [max(init_gesture,init_L),min(end_gesture,end_L)]
                    if intersection[1] >= intersection[0]:
                        distribution[action_dict_gesture[G],action_dict_tools_2_hands[T]] += intersection[1] - intersection[0] +1
                for L_tool in gt_dada_right:
                    T = L_tool[2]
                    if T == "T0":
                        T = "RT0"
                    init_L = int(L_tool[0])
                    end_L =  int(L_tool[1])
                    intersection = [max(init_gesture,init_L),min(end_gesture,end_L)]
                    if intersection[1] >= intersection[0]:
                        distribution[action_dict_gesture[G],action_dict_tools_2_hands[T]] += intersection[1] - intersection[0] +1
    if conditional_on_gesture:
        distribution = distribution / np.expand_dims(np.sum(distribution,axis=1), axis=1 )

    else:
        distribution = distribution/np.sum(distribution)

    ax = sns.heatmap(distribution, annot=True, xticklabels=["no tool in left hand","no tool in right hand","needle driver", "forceps","scissors"], yticklabels=['no gesture', "needle passing" ,"pull the suture", "instrument tie","lay the knot", "cut the suture"], fmt='.3f',cmap=sns.color_palette("mako")
)

    plt.show()


if __name__ == "__main__":
    # dataset_analyzer_gesture("transcriptions_gesture")
    # joint_distribution_analysis("transcriptions_tools_left", "transcriptions_tools_right", "transcriptions_gestures", True)
    save_as_numpy_files(os.path.join("kinematics_with_filtration"),os.path.join("kinematics_with_filtration_npy"))
    # calculate_normalization_parameters_all_folds()
    # save_as_numpy_files(os.path.join("APAS","kinematics_without_filtration"),os.path.join("APAS","kinematics_without_filtration_npy"))

    #http://t-filter.engineerjs.com/
    taps =[  -0.006211464718682453,
      -0.021878614765213328,
      -0.01860344752337795,
      -0.027783297021352776,
      -0.030725028191417213,
      -0.03225564600173213,
      -0.029148656687313576,
      -0.02135734371424111,
      -0.008352421629931162,
      0.009418858067740893,
      0.030948577165696343,
      0.054636075980667914,
      0.07841805221277157,
      0.10004862534725752,
      0.11734070399663442,
      0.1285185684390877,
      0.1323934189340801,
      0.1285185684390877,
      0.11734070399663442,
      0.10004862534725752,
      0.07841805221277157,
      0.054636075980667914,
      0.030948577165696343,
      0.009418858067740893,
      -0.008352421629931162,
      -0.02135734371424111,
      -0.029148656687313576,
      -0.03225564600173213,
      -0.030725028191417213,
      -0.027783297021352776,
      -0.01860344752337795,
      -0.021878614765213328,
      -0.006211464718682453
    ]

    # create_clean_folder("delay_factors.csv",taps=taps,filtration=True)
    # dataset_analyzer_two_hands_tools(os.path.join("APAS","transcriptions_tools_left"),os.path.join("APAS","transcriptions_tools_right"))
    #
    #
    #
    # dataset_analyzer_gesture(os.path.join("transcriptions_gestures"))
