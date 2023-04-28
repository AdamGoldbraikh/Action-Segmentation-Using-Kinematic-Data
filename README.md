# Kinematic Data-Based Action Segmentation for Surgical Applications
This paper is currently under review.

text text
A PyTorch implementation of the paper [Kinematic Data-Based Action Segmentation for Surgical Applications](https://arxiv.org/pdf/2303.07814.pdf).

![Sensors' Locations](figures/sensor_localization_v2.png)

## Install
This implementation uses Python 3.6 and the following packages:
```
opencv-python==4.2.0.32
optuna==2.8.0
numpy==1.19.5
torch==1.8.1
pandas==1.1.5
wandb==0.10.33
tqdm==4.61.2
termcolor==1.1.0
```
We recommend to use conda to deploy the environment

## VTS Dataset
[Data request](https://docs.google.com/forms/d/e/1FAIpQLSeKvalfDwLBkxh1PgrVH14wu2a8UXl7xi0bSAYEU0z9yPrdUA/viewform?usp=sf_link/)


## Run the code
To train and test the model on all the splits run:
```
python train_experiment.py
```
The visualization result is located in `summaries/APAS/experiment_name`,
Where `experiment_name` is a string describing the experiment: the network type, whether it is online, etc.

## Citation
```
@article{goldbraikh2023kinematic,
  title={Kinematic Data-Based Action Segmentation for Surgical Applications},
  author={Goldbraikh, Adam and Shubi, Omer and Rubin, Or and Pugh, Carla M and Laufer, Shlomi},
  journal={arXiv preprint arXiv:2303.07814},
  year={2023}
}```
