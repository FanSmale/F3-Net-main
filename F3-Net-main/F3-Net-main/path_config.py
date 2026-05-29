# -*- coding: utf-8 -*-
"""
Path setting

Created on Feb 2023

@author: Xing-Yi Zhang (zxy20004182@163.com)

"""

from param_config import *                                                      # Get the dataset name
import os

###################################################
####                 PATHS                    #####
###################################################

main_dir        = 'E:/FWI/ddnet-main/'
data_dir        = 'E:\FWI\ddnet-main\data\OpenFWI/'                            #D:/Users/Dora/Documents/Marmousi/ The path of dataset E:\FWI\ddnet-main\data\OpenFWI/  D:/Users/Dora/Microsoft/
results_dir     = main_dir + 'results/'                                         # Output path of run results (not model information)
models_dir      = main_dir + 'models/'                                          # The path where the model will be stored at the end of the run

###################################################
####              DYNAMIC PATHS               #####
###################################################

temp_results_dir= results_dir + '{}Results/'.format(dataset_name)                   # Generate results storage paths for specific dataset
temp_models_dir = models_dir  + '{}Model/'.format(dataset_name)                 # Generate model   storage paths for specific dataset
data_dir        = data_dir    + '{}/'.format(dataset_name)                      # Generate data    storage paths for specific dataset

if os.path.exists(temp_results_dir) and os.path.exists(temp_models_dir):
    results_dir = temp_results_dir
    models_dir  = temp_models_dir
else:
    os.makedirs(temp_results_dir)
    os.makedirs(temp_models_dir)
    results_dir = temp_results_dir
    models_dir  = temp_models_dir

NoiseFlag = False  # If True add noise.
modelName = 'F3'
# VelocityGAN | InversionNet | DD-Net70
# ABA-net | ABA-Loss | ABA-FWI

tagM1 = '_TrainSize' + str(train_size)
tagM2 = '_Epoch' + str(thirdstage_epochs)
tagM3 = '_BatchSize' + str(train_batch_size)
tagM4 = '_LR' + str(learning_rate)

ModelName = modelName + tagM1 + tagM2 + tagM3 + tagM4

# Load pre-trained model
PreModelname = ''