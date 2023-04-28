# parts of the code were adapted from: https://github.com/sj-li/MS-TCN2?utm_source=catalyzex.com
import torch
from Trainer import Trainer
from batch_gen import BatchGenerator
import os
import argparse
import pandas as pd
from datetime import datetime
from termcolor import colored, cprint
import random
import time


dt_string = datetime.now().strftime("%d.%m.%Y %H-%M-%S")


parser = argparse.ArgumentParser()
# # General args

parser.add_argument('--dataset',choices=['VTS', 'BRS','JIGSAWS'], default="VTS")
parser.add_argument('--task',choices=['gestures'], default="gestures")
parser.add_argument('--network',choices=['MS-TCN2','MS-TCN2_ISR','MS-TCN-LSTM','MS-TCN-GRU'], default="MS-TCN-LSTM")
parser.add_argument('--split',choices=['0', '1', '2', '3','4', 'all'], default='all')
parser.add_argument('--eval_rate', default=1, type=int)
parser.add_argument('--offline_mode', default=True, type=bool)
parser.add_argument('--project', default="proj name", type=str)
parser.add_argument('--group', default=dt_string + " ", type=str)
# parser.add_argument('--use_gpu_num',default ="0", type=str )
parser.add_argument('--debagging', default=False, type=bool)
parser.add_argument('--upload', default=False, type=bool)
parser.add_argument('--num_epochs', default=40, type=int)
parser.add_argument('--normalization', choices=['Standard', 'samplewise_SD', 'none'], default='samplewise_SD', type=str)
parser.add_argument('--lefthanded_VTS', default=False, type=bool)

args, unknown = parser.parse_known_args()

if args.dataset == "VTS":
    parser.add_argument('--features_dim', default='36', type=int)
elif args.dataset == "BRS":
    parser.add_argument('--features_dim', default='36', type=int)
elif args.dataset == "JIGSAWS":
    parser.add_argument('--features_dim', default='14', type=int)
else:
    raise NotImplementedError



if args.network == "MS-TCN2":
    parser.add_argument('--lr', default='0.0011', type=float)

    # Architectuyre
    parser.add_argument('--num_f_maps', default='128', type=int)

    parser.add_argument('--num_layers_TCN', default=13, type=int)
    parser.add_argument('--num_R', default=1, type=int)

    parser.add_argument('--hidden_dim_rnn', default=256, type=int)
    parser.add_argument('--num_layers_rnn', default=2, type=int)
    parser.add_argument('--sample_rate', default=1, type=int)
    parser.add_argument('--secondary_sampling', default=1, type=int)


    parser.add_argument('--loss_tau',default=16, type=float)
    parser.add_argument('--loss_lambda', default=0.61, type=float)
    parser.add_argument('--dropout_TCN', default=0.59, type=float)
    parser.add_argument('--dropout_RNN', default=None, type=float)

elif args.network == "MS-TCN-GRU":


    parser.add_argument('--lr', default='0.0017792620184879', type=float)

    # Architectuyre
    parser.add_argument('--num_f_maps', default='256', type=int)

    parser.add_argument('--num_layers_TCN', default=13, type=int)
    parser.add_argument('--num_R', default=1, type=int)

    parser.add_argument('--hidden_dim_rnn', default=256, type=int)
    parser.add_argument('--num_layers_rnn', default=2, type=int)
    parser.add_argument('--sample_rate', default=1, type=int)
    parser.add_argument('--secondary_sampling', default=6, type=int)


    parser.add_argument('--loss_tau',default=16, type=float)
    parser.add_argument('--loss_lambda', default=0.638182400238588, type=float)
    parser.add_argument('--dropout_TCN', default=0.644529097000488, type=float)
    parser.add_argument('--dropout_RNN', default=0.574723886574644, type=float)
    parser.add_argument('--offline_mode', default=True, type=bool)

elif args.network == "MS-TCN-LSTM":

    parser.add_argument('--lr', default='0.0010351748096577', type=float)

    # Architectuyre
    parser.add_argument('--num_f_maps', default='256', type=int)

    parser.add_argument('--num_layers_TCN', default=11, type=int)
    parser.add_argument('--num_R', default=1, type=int)

    parser.add_argument('--hidden_dim_rnn', default=128, type=int)
    parser.add_argument('--num_layers_rnn', default=2, type=int)
    parser.add_argument('--sample_rate', default=2, type=int)
    parser.add_argument('--secondary_sampling', default=3, type=int)


    parser.add_argument('--loss_tau',default=16, type=float)
    parser.add_argument('--loss_lambda', default=0.933391073252775, type=float)
    parser.add_argument('--dropout_TCN', default=0.546245300839604, type=float)
    parser.add_argument('--dropout_RNN', default=0.618678009242687, type=float)




args, unknown = parser.parse_known_args()
augment_dict = {}

# augment_dict = {"reflect":[0.5, 50]}
#
#
# augment_dict = {
#     "rotate_world": [7,1,False],
# }


# augment_dict = {
#     "rotate_world": [7,1,False],
# "reflect":[0.5, 50]
# }

# augment_dict = {"reflect":[0.5, 50]}

debagging = args.debagging
if debagging:
    args.upload = False


print(args)

seed = int(time.time())

random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpu_num

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# use the full temporal resolution @ 30Hz
if args.network in ["GRU","LSTM"]:
    sample_rate = args.sample_rate
    bz = 5

else:
    sample_rate = args.sample_rate
    bz = 2  #2

list_of_splits =[]
if len(args.split) == 1:
    list_of_splits.append(int(args.split))
    if args.dataset in ["VTS", 'ACS']:
        features_dim = 36
    elif args.dataset == "JIGSAWS":
        features_dim = 14


elif args.dataset in ["VTS", 'ACS']:
    list_of_splits = list(range(0,5))
    features_dim = 36

elif args.dataset == "JIGSAWS":
    list_of_splits = list(range(0,8))
    features_dim = 14

else:
    raise NotImplemented
loss_lambda = args.loss_lambda
loss_tau = args.loss_tau
num_epochs = args.num_epochs
eval_rate = args.eval_rate
lr = args.lr
offline_mode = args.offline_mode
num_layers_TCN = args.num_layers_TCN
secondary_sampling = args.secondary_sampling
num_f_maps = args.num_f_maps
experiment_name = args.group+" "+ args.dataset +" task-"  + args.task + " splits- " + args.split +" net- " + args.network +" norm- " +args.normalization +" seed- " + str(seed)
if "rotate_world" in augment_dict:
    experiment_name = experiment_name + " rotate- " + str(augment_dict["rotate_world"][0] ) + " probRot- "  + str(augment_dict["rotate_world"][1]) + " isExtrinsic- " + str(augment_dict["rotate_world"][2])

if "reflect" in augment_dict:
    experiment_name = experiment_name + " reflecProb- " + str(augment_dict["reflect"][0])

if args.num_epochs > 40:
    experiment_name = experiment_name + " num_epochs- " + str(args.num_epochs)

# experiment_name = experiment_name + " ZYX "


if args.lefthanded_VTS:
    experiment_name = experiment_name + " LH"

args.group = experiment_name
hyper_parameter_tuning = False
print(colored(experiment_name, "green"))

summaries_dir = "./summaries/" + args.dataset + "/" + experiment_name
if not debagging:
    if not os.path.exists(summaries_dir):
        os.makedirs(summaries_dir)


full_eval_results = pd.DataFrame()
full_train_results = pd.DataFrame()
full_test_results = pd.DataFrame()


for split_num in list_of_splits:
    if args.dataset == "JIGSAWS":
        seed = 1667815330
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    print("split number: " + str(split_num))
    args.split = str(split_num)
    ToPrint = experiment_name + " split number- " + str(split_num)

    folds_folder = os.path.join("data",args.dataset,"folds")

    if args.dataset in ["VTS", "BRS"]:
        features_path = os.path.join("data",args.dataset,"kinematics_with_filtration_npy")
        mapping_gestures_file = os.path.join("data", args.dataset, "mapping_gestures.txt")
        gt_path_gestures = os.path.join("data", args.dataset,"transcriptions_gestures")


    else:
        # features_path = "./data/" + args.dataset + "/kinematics_npy/"
        features_path = os.path.join("data", args.dataset,"kinematics_npy")

        # gt_path_gestures = "./data/"+args.dataset+"/transcriptions_gestures/"
        gt_path_gestures = os.path.join("data", args.dataset,"transcriptions_gestures")
        # gt_path_tools_left = "./data/"+args.dataset+"/transcriptions_tools_left/"
        gt_path_tools_left= os.path.join("data", args.dataset, "transcriptions_tools_left")
        # gt_path_tools_right = "./data/"+args.dataset+"/transcriptions_tools_right/"
        gt_path_tools_right = os.path.join("data", args.dataset, "transcriptions_tools_right")

        # mapping_gestures_file = "./data/"+args.dataset+"/mapping_gestures.txt"
        mapping_gestures_file = os.path.join("data", args.dataset, "mapping_gestures.txt")

        # mapping_tool_file = "./data/"+args.dataset+"/mapping_tools.txt"
        mapping_tool_file = os.path.join("data", args.dataset, "mapping_tools.txt")


    # model_dir = "./models/"+args.dataset+"/"+ experiment_name+"/split_"+args.split
    model_dir = os.path.join("models", args.dataset,experiment_name,"split" +args.split)

    if not debagging:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    file_ptr = open(mapping_gestures_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict_gestures = dict()
    for a in actions:
        actions_dict_gestures[a.split()[1]] = int(a.split()[0])
    num_classes_tools =0
    actions_dict_tools = dict()

    num_classes_gestures = len(actions_dict_gestures)

    if args.task == "gestures":
        num_classes_list =[num_classes_gestures]
    elif args.dataset == "VTS" and args.task == "tools":
        num_classes_list = [num_classes_tools,num_classes_tools]
    elif args.dataset == "VTS" and args.task == "multi-taks":
        num_classes_list=[num_classes_gestures, num_classes_tools, num_classes_tools]

    trainer = Trainer(num_layers_TCN, num_layers_TCN, args.num_R, num_f_maps, features_dim, num_classes_list,
                      offline_mode=offline_mode,
                      tau=loss_tau, lambd=loss_lambda, hidden_dim_rnn=args.hidden_dim_rnn,
                      num_layers_rnn=args.num_layers_rnn,
                      dropout_TCN=args.dropout_TCN, dropout_RNN=args.dropout_RNN, task=args.task, device=device,
                      network=args.network,
                      secondary_sampling=secondary_sampling,
                      hyper_parameter_tuning=hyper_parameter_tuning,
                      debagging=debagging,ToPrint=ToPrint,dataset=args.dataset)
    print(gt_path_gestures)

    batch_gen = BatchGenerator(num_classes_gestures,num_classes_tools, actions_dict_gestures,actions_dict_tools,features_path,split_num,folds_folder ,gt_path_gestures, "", "", sample_rate=sample_rate,normalization=args.normalization,task=args.task,augment_dict=augment_dict,left_handed=args.lefthanded_VTS,dataset=args.dataset)
    eval_dict ={"features_path":features_path,"actions_dict_gestures": actions_dict_gestures, "actions_dict_tools":actions_dict_tools, "device":device, "sample_rate":sample_rate,"eval_rate":eval_rate,
                "gt_path_gestures":gt_path_gestures, "gt_path_tools_left":"", "gt_path_tools_right":"","task":args.task}
    best_valid_results, eval_results, train_results, test_results = trainer.train(model_dir, batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr,eval_dict=eval_dict,args=args)


    if not debagging:
        eval_results = pd.DataFrame(eval_results)
        train_results = pd.DataFrame(train_results)
        test_results = pd.DataFrame(test_results)
        eval_results["split_num"] = str(split_num)
        train_results["split_num"] = str(split_num)
        test_results["split_num"] = str(split_num)
        eval_results["seed"] = str(seed)
        train_results["seed"] = str(seed)
        test_results["seed"] = str(seed)

        full_eval_results = pd.concat([full_eval_results, eval_results], axis=0)
        full_train_results = pd.concat([full_train_results, train_results], axis=0)
        full_test_results = pd.concat([full_test_results, test_results], axis=0)
        full_eval_results.to_csv(summaries_dir+"/"+args.network +"_evaluation_results.csv",index=False)
        full_train_results.to_csv(summaries_dir+"/"+args.network +"_train_results.csv",index=False)
        full_test_results.to_csv(summaries_dir+"/"+args.network +"_test_results.csv",index=False)



