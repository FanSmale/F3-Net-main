# -*- coding: utf-8 -*-
"""
Parameters setting

Created on Feb 2023

@author: Xing-Yi Zhang (Zhangzxy20004182@163.com)

"""

####################################################
####             MAIN PARAMETERS                ####
####################################################

# Existing datasets: SEGSalt|SEGSimulation|FlatVelA|CurveFaultA|FlatFaultA|CurveVelA)
dataset_name  = 'SEGSalt'
learning_rate = 0.001                               # Learning rate
classes = 1                                         # Number of output channels
display_step = 20000                                    # Number of training sessions required to print a "loss"
SaveEpoch = 115
ReUse = False  # If False always re-train a network
####################################################
####            DATASET PARAMETERS              ####
####################################################

if dataset_name  == 'SEGSimulation':
    data_dim = [400, 301]                           # Dimension of original one-shot seismic data
    model_dim = [201, 301]                          # Dimension of one velocity model
    inchannels = 29                                 # Number of input channels
    train_size = 1600                               # Number of training sets
    test_size = 100                                 # Number of testing sets
    # 验证集大小
    ValSize = 50  # 或者你实际想用的样本数

    firststage_epochs = 0
    secondstage_epochs = 0
    thirdstage_epochs = 400
    loss_weight = [1, 1e6]
    epochs = firststage_epochs + secondstage_epochs + thirdstage_epochs

    train_batch_size = 10                           # Number of batches fed in network in one training epoch.
    test_batch_size = 2

elif dataset_name  == 'SEGSalt':
    data_dim = [400, 301]
    model_dim = [201, 301]
    inchannels = 29
    train_size = 130
    test_size = 10
    ValSize = 5

    firststage_epochs = 0
    secondstage_epochs = 0
    thirdstage_epochs = 200                          # SEGSalt for transfer learning and does not require curriculum tasks
    loss_weight = [1, 1e6]
    epochs = firststage_epochs + secondstage_epochs + thirdstage_epochs

    train_batch_size = 10
    test_batch_size = 2
elif dataset_name == 'marmousi_70_70':
    data_dim = [400, 301]
    model_dim = [70, 70]
    inchannels = 5
    train_size = 30926
    test_size = 328
    ValSize = 328

    firststage_epochs = 10
    secondstage_epochs = 10
    thirdstage_epochs = 140
    loss_weight = [1, 0.01]
    epochs = firststage_epochs + secondstage_epochs + thirdstage_epochs

    train_batch_size = 20
    test_batch_size = 5
elif dataset_name == 'FlatVelB':
    data_dim = [1000, 70]
    model_dim = [70, 70]
    inchannels = 5
    train_size = 24000
    test_size = 6000
    ValSize = 1000

    firststage_epochs = 10
    secondstage_epochs = 10
    thirdstage_epochs = 5
    loss_weight = [1, 0.01]
    epochs = firststage_epochs + secondstage_epochs + thirdstage_epochs

    train_batch_size = 20
    test_batch_size = 5

elif dataset_name == 'CurveVelB':
    data_dim = [1000, 70]
    model_dim = [70, 70]
    inchannels = 5
    train_size = 24000
    ValSize =1000
    test_size =6000

    firststage_epochs = 10
    secondstage_epochs = 10
    thirdstage_epochs = 115
    loss_weight = [1, 0.1]
    epochs = firststage_epochs + secondstage_epochs + thirdstage_epochs

    train_batch_size = 20
    test_batch_size = 5

elif dataset_name == 'FlatFaultA':
    data_dim = [1000, 70]
    model_dim = [70, 70]
    inchannels = 5
    train_size = 48000
    test_size = 6000
    ValSize = 1000
    firststage_epochs = 10
    secondstage_epochs = 10
    thirdstage_epochs = 140
    loss_weight = [1, 0.1]
    epochs = firststage_epochs + secondstage_epochs + thirdstage_epochs

    train_batch_size = 20
    test_batch_size = 5

elif dataset_name == 'CurveFaultB':
    data_dim = [1000, 70]
    model_dim = [70, 70]
    inchannels = 5
    train_size = 48000
    test_size = 6000
    ValSize = 1000

    firststage_epochs = 10
    secondstage_epochs = 10
    thirdstage_epochs = 140
    loss_weight = [1, 0.1]
    epochs = firststage_epochs + secondstage_epochs + thirdstage_epochs

    train_batch_size = 20
    test_batch_size = 5

else:
    print('The selected dataset is invalid')
    exit(0)
