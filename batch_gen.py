#!/usr/bin/python2.7

import torch
import numpy as np
import random
import os
import pandas as pd
from scipy.spatial.transform import Rotation as R
import math
from sklearn.svm import SVC


def reflect_sensor(sensors_data,reflection_mat):
    reflection_mat = np.pad(reflection_mat, [(0, 1), (0, 1)], mode='constant')
    reflection_mat[-1, -1] = 1
    original_point = np.array(sensors_data[0:3,:])
    orig_euler = np.array(sensors_data[3:6,:])
    orig_orientation  = R.from_euler('ZYX', orig_euler.T, degrees=True)
    orig_orientation = orig_orientation.as_matrix()
    new_point = np.matmul(reflection_mat,original_point)
    new_orientation = np.matmul(orig_orientation, reflection_mat)
    orientation_obj = R.from_matrix(new_orientation)

    new_euler = orientation_obj.as_euler('ZYX', degrees=True)
    # new_euler=orig_euler.T
    output = np.concatenate((new_point, new_euler.T), axis=0)
    return output

def reflect_data(features,prob=1,dilute = 50):
    p = random.random()
    if p <= prob:
        sensors = [features[0:6,:], features[6:12, :], features[12:18, :],features[18:24, :]
                    ,features[24:30, :], features[30:36, :]]

        X_left = (sensors[0] + sensors[1] + sensors[2])[0:2, :] / 3
        y_left = np.ones(X_left.shape[1])
        X_right = (sensors[3] + sensors[4] + sensors[5])[0:2, :] / 3
        y_right = np.zeros(X_right.shape[1])
        X_orig = np.concatenate([X_left, X_right], axis=1).T
        y_orig = np.concatenate([y_left, y_right], axis=0)
        X = X_orig[::dilute, :]
        y = y_orig[::dilute]
        svc = SVC(kernel='linear')
        svc.fit(X, y)
        w = svc.coef_[0]
        ## change reprecentation w^T X + c = 0 ==> y= ax + b
        a = -w[0] / w[1]
        b = - (svc.intercept_[0] / w[1])
        theta = math.atan(a)  #theta in radians
        for i,sensor in enumerate(sensors):
            sensors[i][1,:] = sensor[1,:] - b

        ref_mat = np.array([[np.cos(2*theta),np.sin(2*theta)],[np.sin(2*theta),-np.cos(2*theta)]])

        ### apply the reflaction here
        reflected_sensors_123 =[]
        reflected_sensors_456 =[]

        for i,sensor in enumerate(sensors):
            if i<3:
                reflected_sensors_123.append(reflect_sensor(sensor, ref_mat))
            else:
                reflected_sensors_456.append(reflect_sensor(sensor, ref_mat))


        reflected_sensors = reflected_sensors_456 + reflected_sensors_123
        for i,sensor in enumerate(reflected_sensors):
            reflected_sensors[i][1, :] = sensor[1, :] + b

        return np.concatenate(reflected_sensors, axis=0)

    else:
        return features




def rotate_world(sensors_data,rorating_matrix, interpretAsExtrinsicEuler="zyx"):

    original_point = np.array(sensors_data[0:3,:])
    orig_euler = np.array(sensors_data[3:6,:])
    orig_mat = R.from_euler(interpretAsExtrinsicEuler, orig_euler.T, degrees=True)
    new_point = rorating_matrix.apply(original_point.T)
    new_rot_mat = rorating_matrix * orig_mat
    new_euler = new_rot_mat.as_euler(interpretAsExtrinsicEuler, degrees=True)
    new_euler = new_euler
    return np.concatenate((new_point, new_euler), axis=1).T


def RotateWorld(features,theta_max = 7,prob =0.5, interpretAsExtrinsicEuler=True):
    p = random.random()
    if p <= prob:
        if interpretAsExtrinsicEuler:
            interpretAsExtrinsicEuler = "zyx"
        else:
            interpretAsExtrinsicEuler = "ZYX"

        sensors = [features[0:6,:], features[6:12, :], features[12:18, :],features[18:24, :]
                    ,features[24:30, :], features[30:36, :]]

        out_sensors = []

        world_rot_euler_angles = np.array([random.uniform(-theta_max,theta_max),random.uniform(-theta_max,theta_max),
                       random.uniform(-theta_max,theta_max)])

        r = R.from_euler(interpretAsExtrinsicEuler, world_rot_euler_angles, degrees=True)
        for sensor in sensors:
            out_sensors.append(rotate_world(sensor, r, interpretAsExtrinsicEuler))
        return np.concatenate(out_sensors, axis=0)
    else:
        return features





def RotateWorld_JIGSAWS(features,theta_max = 7,prob =0.5, interpretAsExtrinsicEuler=True):
    p = random.random()
    if p <= prob:
        if interpretAsExtrinsicEuler:
            interpretAsExtrinsicEuler = "zyx"
        else:
            interpretAsExtrinsicEuler = "ZYX"

        sensors = [features[0:6,:], features[7:13, :]]
        grippers = [features[6,:], features[13, :]]
        out_sensors = []

        world_rot_euler_angles = np.array([random.uniform(-theta_max,theta_max),random.uniform(-theta_max,theta_max),
                       random.uniform(-theta_max,theta_max)])

        r = R.from_euler(interpretAsExtrinsicEuler, world_rot_euler_angles, degrees=True)
        for sensor,gripper in zip(sensors,grippers):
            out_sensors.append(rotate_world(sensor, r, interpretAsExtrinsicEuler))
            out_sensors.append(np.expand_dims(gripper, axis=0))
        return np.concatenate(out_sensors, axis=0)
    else:
        return features


def reflect_data_JIGSAWS(features,prob=1,dilute = 50):
    p = random.random()
    if p <= prob:

        sensors = [features[0:6,:], features[7:13, :]]
        grippers = [features[6,:], features[13, :]]


        X_left = sensors[0]

        y_left = np.ones(X_left.shape[1])

        X_right = sensors[1]
        y_right = np.zeros(X_right.shape[1])

        X_orig = np.concatenate([X_left, X_right], axis=1).T
        y_orig = np.concatenate([y_left, y_right], axis=0)
        X = X_orig[::dilute, :]
        y = y_orig[::dilute]
        svc = SVC(kernel='linear')
        svc.fit(X, y)
        w = svc.coef_[0]
        ## change reprecentation w^T X + c = 0 ==> y= ax + b
        a = -w[0] / w[1]
        b = - (svc.intercept_[0] / w[1])
        theta = math.atan(a)  #theta in radians
        for i,sensor in enumerate(sensors):
            sensors[i][1,:] = sensor[1,:] - b

        ref_mat = np.array([[np.cos(2*theta),np.sin(2*theta)],[np.sin(2*theta),-np.cos(2*theta)]])

        ### apply the reflaction here
        reflected_sensors_1 =[]
        reflected_sensors_2 =[]

        for i,sensor in enumerate(sensors):
            if i<1:
                reflected_sensors_1.append(reflect_sensor(sensor, ref_mat))
            else:
                reflected_sensors_2.append(reflect_sensor(sensor, ref_mat))


        reflected_sensors = reflected_sensors_1 + reflected_sensors_2
        for i,sensor in enumerate(reflected_sensors):
            reflected_sensors[i][1, :] = sensor[1, :] + b

        output_array =[]
        for i in range(len(reflected_sensors)):
            output_array.append(reflected_sensors[i])
            output_array.append(np.expand_dims(grippers[i], axis=0))
        return np.concatenate(output_array, axis=0)

    else:
        return features


class BatchGenerator(object):
    def __init__(self, num_classes_gestures,num_classes_tools, actions_dict_gestures,actions_dict_tools,features_path,split_num,folds_folder ,gt_path_gestures=None, gt_path_tools_left=None, gt_path_tools_right=None, sample_rate=1,normalization="None",task="gestures",calc_velocity=True,augment_dict={},left_handed=False,dataset=None):
        """
        
        :param num_classes_gestures: 
        :param num_classes_tools: 
        :param actions_dict_gestures: 
        :param actions_dict_tools: 
        :param features_path: 
        :param split_num: 
        :param folds_folder: 
        :param gt_path_gestures: 
        :param gt_path_tools_left: 
        :param gt_path_tools_right: 
        :param sample_rate: 
        :param normalization: None - no normalization, min-max - Min-max feature scaling, Standard - Standard score	 or Z-score Normalization
        ## https://en.wikipedia.org/wiki/Normalization_(statistics)
        """""
        self.dataset = dataset
        self.task = task
        self.normalization = normalization
        self.folds_folder = folds_folder
        self.split_num = split_num
        self.list_of_train_examples = list()
        self.list_of_valid_examples = list()
        self.list_of_test_examples = list()
        self.index = 0
        self.num_classes_gestures = num_classes_gestures
        self.num_classes_tools = num_classes_tools
        self.actions_dict_gestures = actions_dict_gestures
        self.action_dict_tools = actions_dict_tools
        self.gt_path_gestures = gt_path_gestures
        self.gt_path_tools_left = gt_path_tools_left
        self.gt_path_tools_right = gt_path_tools_right
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.velocity = calc_velocity
        self.augment_dict = augment_dict
        self.left_handed = left_handed
        self.read_data()
        if normalization == 'Standard':
            self.normalization_params_read()


    def normalization_params_read(self):
        params = pd.read_csv(os.path.join(self.folds_folder, "std_params_fold_" + str(self.split_num) + ".csv"),index_col=0).values
        self.max = params[0, :]
        self.min = params[1, :]
        self.mean = params[2, :]
        self.std = params[3, :]


    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_train_examples)


    def has_next(self):
        if self.index < len(self.list_of_train_examples):
            return True
        return False


    def read_data(self):
        self.list_of_train_examples =[]
        number_of_folds =0
        for file in os.listdir(self.folds_folder):
            filename = os.fsdecode(file)
            if filename.endswith(".txt") and "test" in filename:
                number_of_folds = number_of_folds + 1

        for file in os.listdir(self.folds_folder):
            filename = os.fsdecode(file)
            if filename.endswith(".txt") and "test" in filename:
                if str(self.split_num) in filename:
                    if self.left_handed == False:
                        file_ptr = open(os.path.join(self.folds_folder, filename), 'r')
                        self.list_of_test_examples = file_ptr.read().split('\n')[:-1]
                        file_ptr.close()
                        random.shuffle(self.list_of_test_examples)
                elif str((self.split_num + 1) % number_of_folds) in filename:
                    file_ptr = open(os.path.join(self.folds_folder, filename), 'r')
                    list_of_examples_fold = file_ptr.read().split('\n')[:-1]

                    file_ptr = open(os.path.join(self.folds_folder, "valid "+str((self.split_num) % number_of_folds) + ".txt"), 'r')


                    self.list_of_valid_examples  =  file_ptr.read().split('\n')[:-1]
                    to_train =[]
                    for element in list_of_examples_fold:
                        if element not in self.list_of_valid_examples:
                            to_train.append(element)

                    random.shuffle(self.list_of_valid_examples)
                    self.list_of_train_examples = self.list_of_train_examples + to_train

                    file_ptr.close()
                else:
                    file_ptr = open(os.path.join(self.folds_folder, filename), 'r')
                    self.list_of_train_examples = self.list_of_train_examples + file_ptr.read().split('\n')[:-1]
                    file_ptr.close()
                continue
            else:
                continue

        random.shuffle(self.list_of_train_examples)
        if self.left_handed:
            self.list_of_test_examples= ["P031_tissue1.csv","P031_tissue2.csv","P031_balloon1.csv","P031_balloon2.csv"]

    def pars_ground_truth(self, gt_source):
        contant = []
        first_frame = None
        for i, line in enumerate(gt_source):
            info = line.split()
            if i == 0:
                first_frame = info[0]
            line_contant = [info[2]] * (int(info[1]) - int(info[0]) + 1)
            contant = contant + line_contant
        return contant, int(first_frame)




##### this is supports one and two heads and 3 heads #############

    def next_batch(self, batch_size):
        batch = self.list_of_train_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target_gestures = []
        batch_target_left = []
        batch_target_right = []

        for seq in batch:
            features = np.load(os.path.join(self.features_path,seq.split('.')[0] + '.npy') )
            if "reflect" in self.augment_dict:
                if self.dataset == "JIGSAWS":
                    features = reflect_data_JIGSAWS(features, prob=self.augment_dict["reflect"][0],
                                            dilute=self.augment_dict["reflect"][1])

                else:
                    features =reflect_data(features, prob=self.augment_dict["reflect"][0], dilute = self.augment_dict["reflect"][1])

            if "rotate_world" in self.augment_dict:
                if self.dataset == "JIGSAWS":
                    features = RotateWorld_JIGSAWS(features, theta_max=self.augment_dict["rotate_world"][0], prob=self.augment_dict["rotate_world"][1], interpretAsExtrinsicEuler=self.augment_dict["rotate_world"][2])
                else:

                    features = RotateWorld(features, theta_max=self.augment_dict["rotate_world"][0], prob=self.augment_dict["rotate_world"][1], interpretAsExtrinsicEuler=self.augment_dict["rotate_world"][2])
            if self.velocity:
                velocity_a = features[:,1:]
                velocity_b = features[:,0:-1]
                velocity = np.zeros_like(features)
                velocity[:,1:] = velocity_a - velocity_b
                features = velocity



            if self.normalization == "Min-max":
                numerator =features.T - self.min
                denominator = self.max-self.min
                features = (numerator / denominator).T
            elif self.normalization == "Standard":
                numerator =features.T - self.mean
                denominator = self.std
                features = (numerator / denominator).T
            elif self.normalization == "samplewise_SD":
                samplewise_meam = features.mean(axis=1)
                samplewise_std = features.std(axis=1)
                numerator =features.T - samplewise_meam
                denominator = samplewise_std
                features = (numerator / denominator).T



            if self.task == "gestures":
                file_ptr = open(os.path.join(self.gt_path_gestures,seq.split('.')[0] + '.txt'),'r')
                gt_source = file_ptr.read().split('\n')[:-1]
                content, first_frame = self.pars_ground_truth(gt_source)
                features = features[:, max(first_frame - 1,0):]
                classes_size = min(np.shape(features)[1], len(content))

                classes = np.zeros(classes_size)
                for i in range(len(classes)):
                    classes[i] = self.actions_dict_gestures[content[i]]
                batch_target_gestures.append(classes[::self.sample_rate])


            elif self.task == "tools":
                file_ptr_right = open(os.path.join(self.gt_path_tools_right, seq.split('.')[0] + '.txt'), 'r')
                gt_source_right = file_ptr_right.read().split('\n')[:-1]
                content_right = self.pars_ground_truth(gt_source_right)
                file_ptr_left = open(os.path.join(self.gt_path_tools_left, seq.split('.')[0] + '.txt'), 'r')
                gt_source_left = file_ptr_left.read().split('\n')[:-1]
                content_left = self.pars_ground_truth(gt_source_left)

                classes_size_right = min(np.shape(features)[1], len(content_left), len(content_right))
                classes_right = np.zeros(classes_size_right)
                for i in range(classes_size_right):
                    classes_right[i] = self.action_dict_tools[content_right[i]]
                batch_target_right.append(classes_right[::self.sample_rate])

                classes_size_left = min(np.shape(features)[1], len(content_left), len(content_right))
                classes_left = np.zeros(classes_size_left)
                for i in range(classes_size_left):
                    classes_left[i] = self.action_dict_tools[content_left[i]]

                batch_target_left.append(classes_left[::self.sample_rate])

            elif self.task == "multi-taks":
                file_ptr = open(os.path.join(self.gt_path_gestures,seq.split('.')[0] + '.txt'), 'r')
                gt_source = file_ptr.read().split('\n')[:-1]
                content = self.pars_ground_truth(gt_source)
                classes_size = min(np.shape(features)[1], len(content))

                classes = np.zeros(classes_size)
                for i in range(len(classes)):
                    classes[i] = self.actions_dict_gestures[content[i]]
                batch_target_gestures.append(classes[::self.sample_rate])

                file_ptr_right = open(os.path.join(self.gt_path_tools_right,seq.split('.')[0] + '.txt'), 'r')
                gt_source_right = file_ptr_right.read().split('\n')[:-1]
                content_right = self.pars_ground_truth(gt_source_right)
                classes_size_right = min(np.shape(features)[1], len(content_right))
                classes_right = np.zeros(classes_size_right)
                for i in range(len(classes_right)):
                    classes_right[i] = self.action_dict_tools[content_right[i]]

                batch_target_right.append(classes_right[::self.sample_rate])

                file_ptr_left = open(os.path.join(self.gt_path_tools_left, seq.split('.')[0] + '.txt'), 'r')
                gt_source_left = file_ptr_left.read().split('\n')[:-1]
                content_left = self.pars_ground_truth(gt_source_left)
                classes_size_left = min(np.shape(features)[1], len(content_left))
                classes_left = np.zeros(classes_size_left)
                for i in range(len(classes_left)):
                    classes_left[i] = self.action_dict_tools[content_left[i]]

                batch_target_left.append(classes_left[::self.sample_rate])
            batch_input.append(features[:, ::self.sample_rate])



        if self.task == "gestures":
            length_of_sequences = list(map(len, batch_target_gestures))
            batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
            batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
            mask = torch.zeros(len(batch_input), self.num_classes_gestures, max(length_of_sequences), dtype=torch.float)
            for i in range(len(batch_input)):
                batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i][:,:batch_input_tensor.shape[2]])
                batch_target_tensor[i, :np.shape(batch_target_gestures[i])[0]] = torch.from_numpy(batch_target_gestures[i])
                mask[i, :, :np.shape(batch_target_gestures[i])[0]] = torch.ones(self.num_classes_gestures, np.shape(batch_target_gestures[i])[0])

            return batch_input_tensor, batch_target_tensor, mask

        elif self.task == "tools":
            length_of_sequences_left = np.expand_dims(np.array( list(map(len, batch_target_left))),1)
            length_of_sequences_right = np.expand_dims(np.array( list(map(len, batch_target_right))),1)

            length_of_sequences = list(np.min(np.concatenate((length_of_sequences_left, length_of_sequences_right),1),1))

            batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0],
                                             max(length_of_sequences), dtype=torch.float)
            batch_target_tensor_left = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long) * (-100)
            batch_target_tensor_right = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long) * (-100)
            mask = torch.zeros(len(batch_target_right), self.num_classes_tools, max(length_of_sequences), dtype=torch.float)


            for i in range(len(batch_input)):
                batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i][:,:batch_input_tensor.shape[2]])
                batch_target_tensor_left[i, :np.shape(batch_target_left[i])[0]] = torch.from_numpy(batch_target_left[i][:batch_target_tensor_left.shape[1]])
                batch_target_tensor_right[i, :np.shape(batch_target_right[i])[0]] = torch.from_numpy(batch_target_right[i][:batch_target_tensor_right.shape[1]])
                mask[i, :, :np.shape(batch_target_right[i])[0]] = torch.ones(self.num_classes_tools, np.shape(batch_target_right[i])[0])



            return batch_input_tensor, batch_target_tensor_left ,batch_target_tensor_right, mask

        elif self.task == "multi-taks":
            length_of_sequences_left = np.expand_dims(np.array( list(map(len, batch_target_left))),1)
            length_of_sequences_right = np.expand_dims(np.array( list(map(len, batch_target_right))),1)
            length_of_sequences_gestures = np.expand_dims(np.array( list(map(len, batch_target_gestures))),1)


            length_of_sequences = list(np.min(np.concatenate((length_of_sequences_left, length_of_sequences_right,length_of_sequences_gestures),1),1))

            batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0],
                                             max(length_of_sequences), dtype=torch.float)

            batch_target_tensor_left = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long) * (-100)
            batch_target_tensor_right = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long) * (-100)
            batch_target_tensor_gestures = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)

            mask_gesture = torch.zeros(len(batch_input), self.num_classes_gestures, max(length_of_sequences), dtype=torch.float)
            mask_tools = torch.zeros(len(batch_input), self.num_classes_tools, max(length_of_sequences), dtype=torch.float)


            for i in range(len(batch_input)):
                batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i][:,:batch_input_tensor.shape[2]])
                batch_target_tensor_left[i, :np.shape(batch_target_left[i])[0]] = torch.from_numpy(batch_target_left[i][:batch_target_tensor_left.shape[1]])
                batch_target_tensor_right[i, :np.shape(batch_target_right[i])[0]] = torch.from_numpy(batch_target_right[i][:batch_target_tensor_right.shape[1]])
                batch_target_tensor_gestures[i, :np.shape(batch_target_gestures[i])[0]] = torch.from_numpy(batch_target_gestures[i][:batch_target_tensor_gestures.shape[1]])
                mask_gesture[i, :, :length_of_sequences[i]] = torch.ones(self.num_classes_gestures, length_of_sequences[i])
                mask_tools[i, :, :length_of_sequences[i]] = torch.ones(self.num_classes_tools, length_of_sequences[i])


            return batch_input_tensor, batch_target_tensor_left ,batch_target_tensor_right,batch_target_tensor_gestures, mask_gesture,mask_tools
    ##### this is supports one and two heads#############

    def next_batch_backup(self, batch_size):
        batch = self.list_of_train_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        batch_target_left = []
        batch_target_right = []

        for seq in batch:
            features = np.load(self.features_path + seq.split('.')[0] + '.npy')
            if self.normalization == "Min-max":
                numerator =features.T - self.min
                denominator = self.max-self.min
                features = (numerator / denominator).T
            elif self.normalization == "Standard":
                numerator =features.T - self.mean
                denominator = self.std
                features = (numerator / denominator).T

            if self.task == "gestures":
                file_ptr = open(self.gt_path_gestures + seq.split('.')[0] + '.txt', 'r')
                gt_source = file_ptr.read().split('\n')[:-1]
                content = self.pars_ground_truth(gt_source)
                classes_size = min(np.shape(features)[1], len(content))

                classes = np.zeros(classes_size)
                for i in range(len(classes)):
                    classes[i] = self.actions_dict_gestures[content[i]]
                batch_input .append(features[:, ::self.sample_rate])
                batch_target.append(classes[::self.sample_rate])


            elif self.task == "tools":
                file_ptr_right = open(self.gt_path_tools_right + seq.split('.')[0] + '.txt', 'r')
                gt_source_right = file_ptr_right.read().split('\n')[:-1]
                content_right = self.pars_ground_truth(gt_source_right)
                classes_size_right = min(np.shape(features)[1], len(content_right))
                classes_right = np.zeros(classes_size_right)
                for i in range(len(classes_right)):
                    classes_right[i] = self.action_dict_tools[content_right[i]]

                batch_input.append(features[:, ::self.sample_rate])
                batch_target_right.append(classes_right[::self.sample_rate])

                file_ptr_left = open(self.gt_path_tools_left + seq.split('.')[0] + '.txt', 'r')
                gt_source_left = file_ptr_left.read().split('\n')[:-1]
                content_left = self.pars_ground_truth(gt_source_left)
                classes_size_left = min(np.shape(features)[1], len(content_left))
                classes_left = np.zeros(classes_size_left)
                for i in range(len(classes_left)):
                    classes_left[i] = self.action_dict_tools[content_left[i]]

                batch_target_left.append(classes_left[::self.sample_rate])

        if self.task == "gestures":
            length_of_sequences = list(map(len, batch_target))
            batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
            batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
            mask = torch.zeros(len(batch_input), self.num_classes_gestures, max(length_of_sequences), dtype=torch.float)
            for i in range(len(batch_input)):
                batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i][:,:batch_input_tensor.shape[2]])
                batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
                mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes_gestures, np.shape(batch_target[i])[0])

            return batch_input_tensor, batch_target_tensor, mask

        elif self.task == "tools":
            length_of_sequences = list(map(len, batch_target_left))
            batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0],
                                             max(length_of_sequences), dtype=torch.float)
            batch_target_tensor_left = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long) * (-100)
            batch_target_tensor_right = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long) * (-100)
            mask = torch.zeros(len(batch_input), self.num_classes_tools, max(length_of_sequences), dtype=torch.float)


            for i in range(len(batch_input)):
                batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
                batch_target_tensor_left[i, :np.shape(batch_target_left[i])[0]] = torch.from_numpy(batch_target_left[i])
                batch_target_tensor_right[i, :np.shape(batch_target_right[i])[0]] = torch.from_numpy(batch_target_right[i])
                mask[i, :, :np.shape(batch_target_right[i])[0]] = torch.ones(self.num_classes_gestures, np.shape(batch_target_right[i])[0])



            return batch_input_tensor, batch_target_tensor_left ,batch_target_tensor_right


    def next_batch_with_gt_tools_as_input(self, batch_size):
        batch = self.list_of_train_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        batch_target_left = []
        batch_target_right = []

        for seq in batch:
            features = np.load(self.features_path + seq.split('.')[0] + '.npy')
            if self.normalization == "Min-max":
                numerator =features.T - self.min
                denominator = self.max-self.min
                features = (numerator / denominator).T
            elif self.normalization == "Standard":
                numerator =features.T - self.mean
                denominator = self.std
                features = (numerator / denominator).T

            file_ptr = open(self.gt_path_gestures + seq.split('.')[0] + '.txt', 'r')
            gt_source = file_ptr.read().split('\n')[:-1]
            content = self.pars_ground_truth(gt_source)
            classes_size = min(np.shape(features)[1], len(content))

            classes = np.zeros(classes_size)
            for i in range(len(classes)):
                classes[i] = self.actions_dict_gestures[content[i]]
            batch_input .append(features[:, ::self.sample_rate])
            batch_target.append(classes[::self.sample_rate])


            file_ptr_right = open(self.gt_path_tools_right + seq.split('.')[0] + '.txt', 'r')
            gt_source_right = file_ptr_right.read().split('\n')[:-1]
            content_right = self.pars_ground_truth(gt_source_right)
            classes_size_right = min(np.shape(features)[1], len(content_right))
            classes_right = np.zeros(classes_size_right)
            for i in range(len(classes_right)):
                classes_right[i] = self.action_dict_tools[content_right[i]]

            batch_target_right.append(classes_right[::self.sample_rate])

            file_ptr_left = open(self.gt_path_tools_left + seq.split('.')[0] + '.txt', 'r')
            gt_source_left = file_ptr_left.read().split('\n')[:-1]
            content_left = self.pars_ground_truth(gt_source_left)
            classes_size_left = min(np.shape(features)[1], len(content_left))
            classes_left = np.zeros(classes_size_left)
            for i in range(len(classes_left)):
                classes_left[i] = self.action_dict_tools[content_left[i]]

            batch_target_left.append(classes_left[::self.sample_rate])

        # for i in range(len(batch_input)):
        #     min_dim = min([batch_target_left[i].size,batch_target_right[i].size, batch_input[i].shape[1]])
        #     batch_target_left[i] = (np.expand_dims(batch_target_left[i][:min_dim], axis=1).T)/ max(self.action_dict_tools.values())
        #     batch_target_right[i] = (np.expand_dims(batch_target_right[i][:min_dim], axis=1).T)/ max(self.action_dict_tools.values())
        #     batch_input[i] = np.concatenate((batch_input[i][:,:min_dim],batch_target_right[i],batch_target_left[i]), axis=0 )

        for i in range(len(batch_input)):
            min_dim = min([batch_target_left[i].size,batch_target_right[i].size, batch_input[i].shape[1]])
            batch_target_left[i] = (np.expand_dims(batch_target_left[i][:min_dim], axis=1).T)/ max(self.action_dict_tools.values())
            batch_target_right[i] = (np.expand_dims(batch_target_right[i][:min_dim], axis=1).T)/ max(self.action_dict_tools.values())
            batch_input[i] = np.concatenate((batch_target_right[i],batch_target_left[i]), axis=0 )

        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        mask = torch.zeros(len(batch_input), self.num_classes_gestures, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes_gestures, np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, mask
