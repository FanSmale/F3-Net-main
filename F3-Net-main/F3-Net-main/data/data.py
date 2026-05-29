# -*- coding: utf-8 -*-
"""
Created on 2024/6/17 7:52

@author: XUQIONG  (xuqiong@swpu.edu.cn)

"""

from torch.utils.data import Dataset, DataLoader, TensorDataset
from math import ceil
# from data.utils import *
# from data.show import *
from func.datasets_reader import *
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import os
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, TensorDataset

class Dataset_openfwi(Dataset):
    '''
       # Load the training data. It includes only seismic_data and vmodel.
       param_seismic_flag: for loading different frequency seismic data. It is designed for domain adaption.
       para_start_num: Start reading from the number of data.
    '''

    def __init__(self, para_data_dir, para_train_size, para_start_num, para_seismic_flag, para_train_or_test):
        print("---------------------------------")
        print("Loading the datasets...")

        data_set, label_set = batch_read_npyfile(para_data_dir, para_start_num, ceil(para_train_size / 500), para_seismic_flag, para_train_or_test)

        ###################################################################################
        #                          Vmodel normalization                                   #
        ###################################################################################
        print("Normalization in progress...")
        # for each sample
        for i in range(data_set.shape[0]):
            vm = label_set[i][0]
            label_set[i][0] = (vm - np.min(vm)) / (np.max(vm) - np.min(vm))

        ###################################################################################
        self.seismic_data = TensorDataset(torch.from_numpy(data_set[:para_train_size, ...]).float())
        self.vmodel = TensorDataset(torch.from_numpy(label_set[:para_train_size, ...]).float())

    def __getitem__(self, index):
        return self.seismic_data[index], self.vmodel[index]

    def __len__(self):
        return len(self.seismic_data)


def batch_read_npyfile(para_dataset_dir,
                       para_start,
                       para_batch_length,
                       para_seismic_flag = "seismic",
                       para_train_or_test = "train"):
    '''
    Batch read seismic gathers and velocity models for .npy file

    :param dataset_dir:             Path to the dataset
    :param start:                   Start reading from the number of data
    :param batch_length:            Starting from the defined first number of data, how long to read
    :param train_or_test:           Whether the read data is used for training or testing ("train" or "test")
    :return:                        a pair: (seismic data, velocity model, contour of velocity model)
                                    Among them, the dimensions of seismic data, velocity model and contour of velocity
                                    model are all (number of read data * 500, channel, height, width)
                                    dataset （500,5,1000,70）  地震数据  未经任何处理
                                    labelset, （500,1,70,70）  速度模型  未经任何处理
                                    edgeset    (500,1,70,70)   二值边缘图像
    '''



    dataset_list = []
    labelset_list = []

    for i in range(para_start, para_start + para_batch_length):

        ##############################
        ##    Load Seismic Data     ##
        ##############################

        # Determine the seismic data path in the dataset
        filename_seis = para_dataset_dir + '{}_data/{}/seismic{}.npy'.format(para_train_or_test, para_seismic_flag, i)
        print("Reading: {}".format(filename_seis))

        data = np.load(filename_seis).astype(np.float32)
        dataset_list.append(data)

        ##############################
        ##    Load Velocity Model   ##
        ##############################

        # Determine the velocity model path in the dataset
        filename_label = para_dataset_dir + '{}_data/vmodel/vmodel{}.npy'.format(para_train_or_test, i)
        print("Reading: {}".format(filename_label))
        label = np.load(filename_label).astype(np.float32)
        labelset_list.append(label)

    dataset = np.concatenate(dataset_list, axis=0)
    labelset = np.concatenate(labelset_list, axis=0)
    return dataset, labelset


class Dataset_openfwi_test(Dataset):
    '''
       # Load the test data. It includes seismic_data, vmodel, and vmodel_max_min.
       # vmodel_max_min is used for inverting normalization.
    '''

    def __init__(self, para_data_dir, para_train_size, para_start_num, para_seismic_flag, para_train_or_test):
        # start_num: 1 for train, 11 for test
        print("---------------------------------")
        print("· Loading the datasets...")

        data_set, label_set = batch_read_npyfile(para_data_dir, para_start_num, ceil(para_train_size / 500), para_seismic_flag, para_train_or_test)
        vmodel_max_min = np.zeros((para_train_size, 2))
        ###################################################################################
        #                          Vmodel normalization                                   #
        ###################################################################################
        print("Normalization in progress...")
        # for each sample
        for i in range(data_set.shape[0]):
            vm = label_set[i][0]
            vmodel_max_min[i, 0] = np.max(vm)
            vmodel_max_min[i, 1] = np.min(vm)
            label_set[i][0] = (vm - np.min(vm)) / (np.max(vm) - np.min(vm))

        # ##################################################################################
        # Training set
        self.seismic_data = TensorDataset(torch.from_numpy(data_set[:para_train_size, ...]).float())
        self.vmodel = TensorDataset(torch.from_numpy(label_set[:para_train_size, ...]).float())
        self.vmodel_max_min = TensorDataset(torch.from_numpy(vmodel_max_min[:para_train_size, ...]).float())

    def __getitem__(self, index):
        return self.seismic_data[index], self.vmodel[index], self.vmodel_max_min[index]

    def __len__(self):
        return len(self.seismic_data)

class DatasetSEG(Dataset):
    def __init__(self, para_data_dir, para_train_size, para_start_num, para_train_or_test):
        """

        :param para_data_dir: 数据集目录
        :param para_train_size: 训练集大小
        :param para_start_num: 起始，1 for train, 1601 for test, 1600+100（模拟数据），130+10（SEG）
        :param para_train_or_test: 训练or测试
        """
        print("-------------------------------------------------")
        print("Loading the SEG salt datasets...")

        # ceil（）向上取整
        # 批量读取npy文件
        data_set, label_set = batch_read_matfile(para_data_dir, para_train_size, para_start_num, para_train_or_test)

        ###################################################################################
        #                          Vmodel Normalization                                   #
        ###################################################################################
        # print("Vmodel Normalization in progress...")
        # 最小-最大归一化适用于数据分布有明显边界的情况
        # 遍历每个数据，从0-(train_size-1)，将速度模型归一化到[0,1]
        # data_set.shape[0]取第0个值，即train_size, 总共有多少个数据
        # for i in range(data_set.shape[0]):
        #     vm = label_set[i][0]
        #     label_set[i][0] = (vm - np.min(vm)) / (np.max(vm) - np.min(vm))

        # 标准化（Z-score归一化）：将数据转换为均值为0，标准差为1的分布
        # for i in range(data_set.shape[0]):
        #     vm = label_set[i][0]
        #     # 计算均值和标准差
        #     mean_val = np.mean(vm)
        #     std_val = np.std(vm)
        #     # 标准化：(x - mean) / std
        #     label_set[i][0] = (vm - mean_val) / (std_val + 1e-8)  # 添加小常数避免除零
        ###################################################################################
        # torch.from_numpy()可以将NumPy数组直接转换为PyTorch张量，同时保持两者共享底层数据内存的特性。
        # 这意味着对转换后的PyTorch张量进行的任何修改都会反映到原始的NumPy数组上。
        # TensorDataset用于将张量组合成数据集的工具类。它的作用是将多个张量作为输入，构建一个能够以样本对的形式提供数据的数据集。
        self.seismic_data = TensorDataset(torch.from_numpy(data_set[:para_train_size, ...]).float())
        self.vmodel = TensorDataset(torch.from_numpy(label_set[:para_train_size, ...]).float())

        # 打印信息
        print(f"✅ Dataset loaded successfully!")
        print(f"  - Dataset type: {para_train_or_test}")
        print(f"  - Number of samples: {len(self.seismic_data)}")
        print(f"  - Seismic data shape : {data_set.shape}")
        print(f"  - Velocity model shape: {label_set.shape}")
        print("---------------------------------")

    def __getitem__(self, index):
        # ⚠️ TensorDataset返回的是tuple，因此需要取第一个元素
        seis = self.seismic_data[index][0]  # [29, H, W] or [1, 29, H, W]
        vmodel = self.vmodel[index][0]  # [1, H, W]

        # 如果多了一层 1，就去掉
        if seis.dim() == 4 and seis.shape[0] == 1:
            seis = seis.squeeze(0)
        if vmodel.dim() == 4 and vmodel.shape[0] == 1:
            vmodel = vmodel.squeeze(0)

        return seis, vmodel

    def __len__(self):
        return len(self.seismic_data)


def batch_read_matfile(para_dataset_dir,
                       para_batch_length,
                       para_start,
                       para_train_or_test="train",
                       para_inchannels=29):
    """

    :param para_dataset_dir: Path to the dataset
    :param para_batch_length:
    :param para_start:
    :param para_train_or_test:
    :param para_channels:
    :return:
    """

    # {ndarray:(batchsize, 29, 400, 301)}
    data_set = np.zeros([para_batch_length, para_inchannels, data_dim[0], data_dim[1]])
    # {ndarray:(batchsize, 1, 201, 301)}
    label_set = np.zeros([para_batch_length, classes, model_dim[0], model_dim[1]])
    # clabel_set = np.zeros([para_batch_length, OutChannel, ModelDim[0], ModelDim[1]])


    # 索引-index:0  值-i:1
    for indx, i in enumerate(range(para_start, para_batch_length + para_start)):

        # Load Seismic Data
        # SimulateData：1600的模拟数据用于训练，1601-1700用于测试
        # SEGSaltData：130的真实数据用于训练，1-10用于测试
        filename_seis = para_dataset_dir + '{}_data/seismic/seismic{}.mat'.format(para_train_or_test, i)
        print("Reading: {}".format(filename_seis))

        # {ndarray:(400,301,29)}
        sei_data = scipy.io.loadmat(filename_seis)["Rec"]

        ## (400, 301, 29) -> (29, 400, 301)

        # {ndarray:(29,301,400)}
        sei_data = sei_data.swapaxes(0, 2)
        # {ndarray:(29,400,301)}
        sei_data = sei_data.swapaxes(1, 2)

        # index第几个数据  ch第几个通道数
        # data_set 四维（index，ch，400，301）
        for ch in range(para_inchannels):
            data_set[indx, ch, ...] = sei_data[ch, ...]

        # # Load Velocity Model
        # # SimulateData
        # filename_label = para_dataset_dir + '{}_data/vmodel/vmodel{}.mat'.format(para_train_or_test, i)
        # # SEGSaltData
        # # filename_label = para_dataset_dir + '{}_data/vmodel/svmodel{}.mat'.format(para_train_or_test, i)
        # print("Reading: {}".format(filename_label))
        #
        # vm_data = scipy.io.loadmat(filename_label)['vmodel']  # SimulateData:vmodel; SEGSaltData:svmodel
        # -------------------- Load Velocity Model --------------------
        # 自动判断数据集类型，根据路径或数据集名称选择文件前缀和键名
        if "SEGSalt" in para_dataset_dir:
            # SEGSalt 数据
            filename_label = os.path.join(
                para_dataset_dir, f"{para_train_or_test}_data", "vmodel", f"svmodel{i}.mat"
            )
            vmodel_key = "svmodel"
        else:
            # Simulation / SEGSimulation 数据
            filename_label = os.path.join(
                para_dataset_dir, f"{para_train_or_test}_data", "vmodel", f"vmodel{i}.mat"
            )
            vmodel_key = "vmodel"

        print(f"Reading: {filename_label}")

        # 读取 .mat 文件并取对应键
        vm_data = scipy.io.loadmat(filename_label)[vmodel_key]

        # print(vm_data)
        label_set[indx, 0, ...] = vm_data
        #clabel_set[indx, 0, ...] = extract_contours(vm_data)


    return data_set, label_set
    # return data_set, [label_set, label_set]


class Dataset_SEG_Test(Dataset):
    """
    SEG 测试集数据读取类

    返回:
        seismic_data: 地震数据, Tensor, (N, C, H, W)，逐样本标准化
        vmodel: 速度模型, Tensor, (N, 1, H, W)，保持物理尺度（不归一化）
    """

    def __init__(self, dataset_dir, test_size, start_num=1, data_channels=29, train_or_test="test"):
        print("---------------------------------")
        print("· Loading SEG test datasets...")

        # 固定读取模式为 test
        read_mode = "test"

        # === 1. 读取数据 ===
        data_set, [label_set, _] = batch_read_matfile(
            dataset_dir,
            start=start_num,
            batch_length=test_size,
            train_or_test=read_mode,
            data_channels=data_channels
        )

        # === 2. 对地震数据进行标准化（逐样本）===
        print("· Normalizing seismic data (per sample)...")
        for i in range(data_set.shape[0]):
            d = data_set[i]
            mean, std = np.mean(d), np.std(d)
            if std > 1e-6:
                data_set[i] = (d - mean) / std
            else:
                data_set[i] = d - mean

        # === 3. 不归一化速度模型，保持物理尺度 ===
        print("· Keeping velocity models unnormalized.")
        vmin, vmax = np.min(label_set), np.max(label_set)
        print(f"Velocity range in test set: {vmin:.1f} ~ {vmax:.1f} m/s")

        # === 4. 转为 Tensor ===
        self.seismic_data = torch.from_numpy(data_set).float()
        self.vmodel = torch.from_numpy(label_set).float()

    def __getitem__(self, index):
        return self.seismic_data[index], self.vmodel[index]

    def __len__(self):
        return len(self.seismic_data)




class Dataset_openfwi4_test(Dataset):
    '''
       # Load the test data including velocity model edge.
       # It includes seismic_data, vmodel, edge, and vmodel_max_min.
       # vmodel_max_min is used for inverting normalization.
    '''

    def __init__(self, para_data_dir, para_train_size, para_start_num, para_seismic_flag, para_train_or_test):
        # start_num: 1 for train, 11 for test
        print("---------------------------------")
        print("· Loading the datasets...")

        data_set, label_set, edge_set = batch_read_npyfile4(para_data_dir, para_start_num, ceil(para_train_size / 500), para_seismic_flag, para_train_or_test)
        vmodel_max_min = np.zeros((para_train_size, 2))
        ###################################################################################
        #                          Vmodel normalization                                   #
        ###################################################################################
        print("Normalization in progress...")
        # for each sample
        for i in range(data_set.shape[0]):
            vm = label_set[i][0]
            vmodel_max_min[i, 0] = np.max(vm)
            vmodel_max_min[i, 1] = np.min(vm)
            label_set[i][0] = (vm - np.min(vm)) / (np.max(vm) - np.min(vm))

        # ##################################################################################
        # Training set
        self.seismic_data = TensorDataset(torch.from_numpy(data_set[:para_train_size, ...]).float())
        self.vmodel = TensorDataset(torch.from_numpy(label_set[:para_train_size, ...]).float())
        self.edge = TensorDataset(torch.from_numpy(edge_set[:para_train_size, ...]).to(torch.uint8))
        self.vmodel_max_min = TensorDataset(torch.from_numpy(vmodel_max_min[:para_train_size, ...]).float())

    def __getitem__(self, index):
        return self.seismic_data[index], self.vmodel[index], self.edge[index], self.vmodel_max_min[index]

    def __len__(self):
        return len(self.seismic_data)


class Dataset_openfwi4(Dataset):
    '''
       Load the training data including velocity model edge. It includes seismic_data, vmodel, and edge of vmodel.
    '''

    def __init__(self, para_data_dir, para_train_size, para_start_num, para_seismic_flag, para_train_or_test):
        # para_start_num: 1 for train, 11 for test
        print("---------------------------------")
        print("· Loading the datasets...")

        data_set, label_set, edge_set = batch_read_npyfile4(para_data_dir, para_start_num, ceil(para_train_size / 500), para_seismic_flag, para_train_or_test)

        ###################################################################################
        #                          Vmodel normalization                                   #
        ###################################################################################
        print("Normalization in progress...")
        # for each sample
        for i in range(data_set.shape[0]):
            vm = label_set[i][0]
            label_set[i][0] = (vm - np.min(vm)) / (np.max(vm) - np.min(vm))

        # Training set
        self.seismic_data = TensorDataset(torch.from_numpy(data_set[:para_train_size, ...]).float())
        self.vmodel = TensorDataset(torch.from_numpy(label_set[:para_train_size, ...]).float())
        self.edge = TensorDataset(torch.from_numpy(edge_set[:para_train_size, ...]).uint8())

    def __getitem__(self, index):
        return self.seismic_data[index], self.vmodel[index], self.edge[index]

    def __len__(self):
        return len(self.seismic_data)


def batch_read_npyfile4(para_dataset_dir,
                       para_start,
                       para_batch_length,
                       para_seismic_flag = "seismic",
                       para_train_or_test = "train"):
    '''
    Batch read seismic gathers and velocity models for .npy file
    including velocity model edge.

    :param dataset_dir:             Path to the dataset
    :param start:                   Start reading from the number of data
    :param batch_length:            Starting from the defined first number of data, how long to read
    :param train_or_test:           Whether the read data is used for training or testing ("train" or "test")
    :return:                        a pair: (seismic data, velocity model, contour of velocity model)
                                    Among them, the dimensions of seismic data, velocity model and contour of velocity
                                    model are all (number of read data * 500, channel, height, width)
                                    dataset （500,5,1000,70）  地震数据  未经任何处理
                                    labelset, （500,1,70,70）  速度模型  未经任何处理
                                    edgeset    (500,1,70,70)   二值边缘图像
    '''

    dataset_list = []
    labelset_list = []
    edgeset_list = []

    for i in range(para_start, para_start + para_batch_length):
        ##############################
        ##    Load Seismic Data     ##
        ##############################

        # Determine the seismic data path in the dataset
        filename_seis = para_dataset_dir + '{}_data/{}/seismic{}.npy'.format(para_train_or_test, para_seismic_flag, i)
        print("Reading: {}".format(filename_seis))

        datas = np.load(filename_seis).astype(np.float32)
        dataset_list.append(datas)

        ##############################
        ##    Load Velocity Model   ##
        ##############################

        # Determine the velocity model path in the dataset
        filename_label = para_dataset_dir + '{}_data/vmodel/vmodel{}.npy'.format(para_train_or_test, i)
        print("Reading: {}".format(filename_label))
        labels = np.load(filename_label).astype(np.float32)
        labelset_list.append(labels)

        ###################################
        ##    Generating Velocity Edge   ##
        ###################################

        print("Generating velocity model profile......")
        edges = np.zeros([500, 1, model_dim[0], model_dim[1]])
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                edges[i, j, ...] = extract_contours(labels[i, j, ...])
        edgeset_list.append(edges.astype(np.uint8))

    dataset = np.concatenate(dataset_list, axis=0)
    labelset = np.concatenate(labelset_list, axis=0)
    edgeset = np.concatenate(edgeset_list, axis=0)

    return dataset, labelset, edgeset



