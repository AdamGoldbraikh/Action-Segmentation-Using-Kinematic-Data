import numpy as np
import pandas as pd
import os
from data_cleaner import pars_ground_truth
import matplotlib.pyplot as plt
import seaborn as sns

def dataset_analyzer(task):
    if task == "Gestures":
        gt_path ="transcriptions_gesture"
        action_dict = {"G0": 0, "G1": 1, "G2": 2, "G3": 3, "G4": 4, "G5": 5}
        actions_names= ['No gesture', "Needle passing" ,"Pull the suture", "Instrument tie","Lay the knot", "Cut the suture"]
        duration_per_activity = [[], [], [], [], [], []]
    elif task == "Tool usage in left hand":
        gt_path ="transcriptions_tools_left"
        action_dict = {"T0": 0, "T1": 1, "T2": 2, "T3": 3}
        actions_names= ["No tool in hand","Needle driver", "Forceps","Scissors"]
        duration_per_activity = [[], [], [], []]

    elif task == "Tool usage in right hand":
        gt_path ="transcriptions_tools_right"
        action_dict = {"T0": 0, "T1": 1, "T2": 2, "T3": 3}
        actions_names= ["No tool in hand","Needle driver", "Forceps","Scissors"]
        duration_per_activity = [[], [], [], []]

    else:
        raise NotImplemented
    num_of_samples_list = [0]*len(action_dict)
    if task == "Gestures":
        rot = 25
    else:
        rot =0

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
    num_of_samples_lists = pd.DataFrame(num_of_samples_lists,columns=actions_names)
    duration_per_activity = pd.DataFrame(duration_per_activity).transpose() / 30
    duration_per_activity.columns = actions_names

    # duration = duration_per_activity.boxplot(rot=65,grid=False,)
    duration = sns.boxplot( data=duration_per_activity,palette="Set2")
    plt.ylabel("Duration [sec]")
    plt.xticks(rotation=rot)

    if task == "Gestures":
        duration.set_title("The Duration of a Gesture")
    elif task == "Tool usage in left hand":
        duration.set_title("The Duration of Tool Usage in The Left Hand")
    else:
        duration.set_title("The Duration of Tool Usage in The Right Hand")
    plt.show()

    number_of_occurances = sns.boxplot( data=num_of_samples_lists,palette="Set2")
    plt.ylabel("Number of Occurrence")
    plt.xticks(rotation=rot)


    if task == "Gestures":
        number_of_occurances.set_title("The Number of Times a Gesture Appears in a Simulation Session")
    elif task == "Tool usage in left hand":
        number_of_occurances.set_title("The Number of Times a Tool Was Used With the Left Hand \n During the Simulation")
    else:
        number_of_occurances.set_title("The Number of Times a Tool Was Used With the Right Hand \n During the Simulation")


    plt.show()


def joint_distribution_analysis(gt_path_tool,gt_path_gesture,conditional_on_gesture):
    action_dict_gesture = {"G0": 0, "G1": 1, "G2": 2, "G3": 3, "G4": 4, "G5": 5}
    action_dict_tools = {"T0": 0, "T1": 1, "T2": 2, "T3": 3}
    distribution= np.zeros((len(action_dict_gesture),len(action_dict_tools)))
    for file in os.listdir(gt_path_gesture):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            file_ptr = open(os.path.join(gt_path_gesture,filename), 'r')
            gt_dada_gesture = pars_ground_truth(file_ptr)
            file_ptr = open(os.path.join(gt_path_tool,filename), 'r')
            gt_dada_tool = pars_ground_truth(file_ptr)
            # last_index = min(gt_dada_gesture[-1][1],gt_dada_left[-1][1],gt_dada_right[-1][1])
            for gesture in gt_dada_gesture:
                G = gesture[2]
                init_gesture = int(gesture[0])
                end_gesture = int(gesture[1])
                for tool in gt_dada_tool:
                    T= tool[2]
                    init_tool = int(tool[0])
                    end_tool = int(tool[1])
                    intersection = [max(init_gesture,init_tool),min(end_gesture,end_tool)]
                    if intersection[1] >= intersection[0]:
                        distribution[action_dict_gesture[G],action_dict_tools[T]] += intersection[1] - intersection[0] +1
    if conditional_on_gesture:
        distribution = distribution / np.expand_dims(np.sum(distribution,axis=1), axis=1 )

    else:
        distribution = distribution/np.sum(distribution)

    ax = sns.heatmap(distribution, annot=True, xticklabels=["No tool in hand","Needle driver", "Forceps","Scissors"], yticklabels=['No gesture', "Needle passing" ,"Pull the suture", "Instrument tie","Lay the knot", "Cut the suture"], fmt='.3f',cmap=sns.color_palette("mako"))
    plt.xticks(rotation=12)
    if gt_path_tool == "transcriptions_tools_right" and conditional_on_gesture:
        ax.set_title("The Conditional Distribution of Tool Usage in Right Hand Given Gesture")
    elif gt_path_tool == "transcriptions_tools_right" and not conditional_on_gesture:
        ax.set_title("Tool Usage in Right Hand - Gesture Joint Distribution")
    elif gt_path_tool == "transcriptions_tools_left" and conditional_on_gesture:
        ax.set_title("The Conditional Distribution of Tool Usage in Left Hand Given Gesture")
    elif gt_path_tool == "transcriptions_tools_left" and not conditional_on_gesture:
        ax.set_title("Tool Usage in Left Hand - Gesture Joint Distribution")


    plt.show()




if __name__ == "__main__":
    #
    # joint_distribution_analysis("transcriptions_tools_right", "transcriptions_gesture", False)
    joint_distribution_analysis("transcriptions_tools_left", "transcriptions_gesture", False)

    # dataset_analyzer("Gestures")
    # dataset_analyzer("Tool usage in left hand")
    # dataset_analyzer("Tool usage in right hand")

