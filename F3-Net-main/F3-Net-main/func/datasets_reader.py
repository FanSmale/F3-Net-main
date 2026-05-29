# -*- coding: utf-8 -*-
"""
Direct method for reading datasets

Created on Sep 2023

@author: Xing-Yi Zhang (zxy20004182@163.com)

"""
import os

from param_config import *
import scipy.io
import scipy
import numpy as np
from func.utils import extract_contours

def batch_read_matfile(dataset_dir,
                       start,
                       batch_length,
                       train_or_test="train",
                       data_channels=29):
    '''
    Batch read seismic gathers and velocity models for .mat file

    :param dataset_dir:             Path to the dataset
    :param start:                   Start reading from the number of data
    :param batch_length:            Starting from the defined first number of data, how long to read
    :param train_or_test:           Whether the read data is used for training or testing ("train" or "test")
    :param data_channels:           The total number of channels read into the data itself
    :return:                        a quadruple: (seismic data, [velocity model, contour of velocity model])
                                    Among them, the dimensions of seismic data, velocity model and contour of velocity model are all (number of read data, channel, width x height)
    '''

    data_set = np.zeros([batch_length, data_channels, data_dim[0], data_dim[1]])
    label_set = np.zeros([batch_length, classes, model_dim[0], model_dim[1]])
    clabel_set = np.zeros([batch_length, classes, model_dim[0], model_dim[1]])

    for indx, i in enumerate(range(start, start + batch_length)):

        # Load Seismic Data
        # filename_seis = dataset_dir + '{}_data/seismic/seismic{}.mat'.format(train_or_test, i)
        filename_seis = os.path.join(dataset_dir, f"{train_or_test}_data", "seismic", f"seismic{i}.mat")

        print("Reading: {}".format(filename_seis))
        sei_data = scipy.io.loadmat(filename_seis)["Rec"]
        # (400, 301, 29) -> (29, 400, 301)
        sei_data = sei_data.swapaxes(0, 2)
        sei_data = sei_data.swapaxes(1, 2)
        for ch in range(inchannels):
            data_set[indx, ch, ...] = sei_data[ch, ...]

        # Load Velocity Model
        # filename_label = dataset_dir + '{}_data/vmodel/vmodel{}.mat'.format(train_or_test, i)
        # filename_label = os.path.join(dataset_dir, f"{train_or_test}_data", "vmodel", f"svmodel{i}.mat")
        if "SEGSalt" in dataset_dir:
            filename_label = os.path.join(dataset_dir, f"{train_or_test}_data", "vmodel", f"svmodel{i}.mat")
        else:
            filename_label = os.path.join(dataset_dir, f"{train_or_test}_data", "vmodel", f"vmodel{i}.mat")

        print("Reading: {}".format(filename_label))
        if "SEGSalt" in dataset_dir:
            vm_data = scipy.io.loadmat(filename_label)["svmodel"]
        else:
            vm_data = scipy.io.loadmat(filename_label)["vmodel"]

        label_set[indx, 0, ...] = vm_data
        clabel_set[indx, 0, ...] = extract_contours(vm_data)

    return data_set, [label_set, clabel_set]


def batch_read_npyfile(dataset_dir, start, batch_length, train_or_test="train"):
    '''
    批量读取地震数据和速度模型（支持最后一个文件样本不足500条）
    自动推断 classes 和 model_dim
    '''
    dataset = []
    labelset = []

    for i in range(start, start + batch_length):
        # 加载地震数据
        filename_seis = f"{dataset_dir}{train_or_test}_data/seismic/seismic{i}.npy"
        print(f"Reading: {filename_seis}")
        temp_seis = np.load(filename_seis)
        dataset.append(temp_seis)

        # 加载速度模型
        filename_label = f"{dataset_dir}{train_or_test}_data/vmodel/vmodel{i}.npy"
        print(f"Reading: {filename_label}")
        temp_vmod = np.load(filename_label)

        # 检查样本数是否一致
        if temp_seis.shape[0] != temp_vmod.shape[0]:
            print(f"❌ ERROR: Size mismatch at file {i}:")
            print(f"    seismic{i}.npy shape = {temp_seis.shape}")
            print(f"    vmodel{i}.npy  shape = {temp_vmod.shape}")
        else:
            print(f"✅ File {i} OK: {temp_seis.shape[0]} samples")

        labelset.append(temp_vmod)

    # 允许最后一个文件不足500条，使用 concatenate 拼接
    dataset = np.concatenate(dataset, axis=0)
    labelset = np.concatenate(labelset, axis=0)

    total_samples = dataset.shape[0]  # 实际总样本数

    # 自动推断 classes 和 model_dim
    classes = labelset.shape[1]
    model_dim = (labelset.shape[2], labelset.shape[3])

    print(f"Total samples read: {total_samples}")
    print(f"Detected classes: {classes}, model_dim: {model_dim}")
    print("Generating velocity model contour profiles...")

    # 生成轮廓标签
    conlabels = np.zeros([total_samples, classes, model_dim[0], model_dim[1]])
    for idx in range(total_samples):
        for c in range(classes):
            conlabels[idx, c, ...] = extract_contours(labelset[idx, c, ...])

    return dataset, [labelset, conlabels]

# 500个为一组
# def batch_read_npyfile(dataset_dir,
#                        start,
#                        batch_length,
#                        train_or_test="train"):
#     '''
#     Batch read seismic gathers and velocity models for .npy file
#
#     :param dataset_dir:             Path to the dataset
#     :param start:                   Start reading from the number of data
#     :param batch_length:            Starting from the defined first number of data, how long to read
#     :param train_or_test:           Whether the read data is used for training or testing ("train" or "test")
#     :return:                        a pair: (seismic data, [velocity model, contour of velocity model])
#                                     Among them, the dimensions of seismic data, velocity model and contour of velocity
#                                     model are all (number of read data * 500, channel, height, width)
#     '''
#
#     dataset = []
#     labelset = []
#
#     for i in range(start, start + batch_length):
#
#         ##############################
#         ##    Load Seismic Data     ##
#         ##############################
#
#         # Determine the seismic data path in the dataset
#         filename_seis = dataset_dir + '{}_data/seismic/seismic{}.npy'.format(train_or_test, i)
#         print("Reading: {}".format(filename_seis))
#         temp_seis = np.load(filename_seis)
#
#         dataset.append(temp_seis)
#
#         ##############################
#         ##    Load Velocity Model   ##
#         ##############################
#
#         # Determine the velocity model path in the dataset
#         filename_label = dataset_dir + '{}_data/vmodel/vmodel{}.npy'.format(train_or_test, i)
#         print("Reading: {}".format(filename_label))
#         temp_vmod = np.load(filename_label)
#
#         # Check for mismatch
#         if temp_seis.shape[0] != temp_vmod.shape[0]:
#             print(f"❌ ERROR: Size mismatch at file {i}:")
#             print(f"    seismic{i}.npy shape = {temp_seis.shape}")
#             print(f"    vmodel{i}.npy  shape = {temp_vmod.shape}")
#         else:
#             print(f"✅ File {i} OK: {temp_seis.shape[0]} samples")
#         labelset.append(temp_vmod)
#
#     dataset = np.vstack(dataset)
#     labelset = np.vstack(labelset)
#
#     print("Generating velocity model profile......")
#     conlabels = np.zeros([batch_length * 500, classes, model_dim[0], model_dim[1]])
#     for i in range(labelset.shape[0]):
#         for j in range(labelset.shape[1]):
#             conlabels[i, j, ...] = extract_contours(labelset[i, j, ...])
#
#     return dataset, [labelset, conlabels]


def single_read_matfile(dataset_dir,
                        seismic_data_size,
                        velocity_model_size,
                        readID,
                        train_or_test = "train",
                        data_channels = 29):
    '''
    Single read seismic gathers and velocity models for .mat file

    :param dataset_dir:             Path to the dataset
    :param seismic_data_size:       Size of the seimic data
    :param velocity_model_size:     Size of the velocity model
    :param readID:                  The ID number of the selected data
    :param train_or_test:           Whether the read data is used for training or testing ("train" or "test")
    :param data_channels:           The total number of channels read into the data itself
    :return:                        a triplet: (seismic data, velocity model, contour of velocity model)
                                    Among them, the dimensions of seismic data, velocity model and contour of velocity model are
                                    (channel, width, height), (width, height) and (width, height) respectively
    '''
    filename_seis = dataset_dir + '{}_data/seismic/seismic{}.mat'.format(train_or_test, readID)
    print("Reading: {}".format(filename_seis))
    filename_label = dataset_dir + '{}_data/vmodel/svmodel{}.mat'.format(train_or_test, readID)
    print("Reading: {}".format(filename_label))

    se_data = scipy.io.loadmat(filename_seis)
    print(">> MAT file keys:", se_data.keys())

    se_data = np.float32(se_data["Rec"].reshape([seismic_data_size[0], seismic_data_size[1], data_channels]))
    vm_data = scipy.io.loadmat(filename_label)
    print(">> VEL file keys:", vm_data.keys())

    vm_data = np.float32(vm_data["svmodel"].reshape(velocity_model_size[0], velocity_model_size[1]))

    # (400, 301, 29) -> (29, 400, 301)
    se_data = se_data.swapaxes(0, 2)
    se_data = se_data.swapaxes(1, 2)

    contours_vm_data = extract_contours(vm_data)  # Use Canny to extract contour features

    return se_data, vm_data, contours_vm_data

def single_read_npyfile(dataset_dir,
                        readIDs,
                        train_or_test = "train"):
    '''
    Single read seismic gathers and velocity models for .npy file

    :param dataset_dir:             Path to the dataset
    :param readID:                  The IDs number of the selected data
    :param train_or_test:           Whether the read data is used for training or testing ("train" or "test")
    :return:                        seismic data, velocity model, contour of velocity model
    '''

    # Determine the seismic data path in the dataset
    filename_seis = dataset_dir + '{}_data/seismic/seismic{}.npy'.format(train_or_test, readIDs[0])
    print("Reading: {}".format(filename_seis))
    # Determine the velocity model path in the dataset
    filename_label = dataset_dir + '{}_data/vmodel/vmodel{}.npy'.format(train_or_test, readIDs[0])
    print("Reading: {}".format(filename_label))

    se_data = np.load(filename_seis)[readIDs[1]]
    vm_data = np.load(filename_label)[readIDs[1]][0]

    print("Generating velocity model profile......")
    conlabel = extract_contours(vm_data)

    return se_data, vm_data, conlabel
