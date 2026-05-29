# -*- coding: utf-8 -*-
"""
Test the model effect after training.

Created on Sep 2023

@author: Xing-Yi Zhang (Zhangzxy20004182@163.com)

"""
from data.data import *
from data.show import *
from path_config import *
from func.utils import run_mse, run_mae, run_lpips, run_uqi, pain_seg_seismic_data, pain_seg_velocity_model, \
    pain_openfwi_velocity_model, pain_openfwi_seismic_data, local_MSE,local_MAE,add_gasuss_noise, magnify_amplitude_fornumpy, SSIM,PSNR
from func.datasets_reader import batch_read_matfile, batch_read_npyfile, single_read_matfile, single_read_npyfile
from model_train import determine_network
import matplotlib.pyplot as plt
import time
import lpips
import numpy as np
import torch
import torch.utils.data as data_utils
import glob
import matplotlib
matplotlib.use('TkAgg')


def load_dataset():
    '''
    Load the testing data according to the parameters in "param_config"

    :return:    A triplet: datasets loader, seismic gathers and velocity models
    '''

    print("---------------------------------")
    print("· Loading the datasets...")
    if dataset_name in ['SEGSalt', 'SEGSimulation']:
        data_set, label_sets = batch_read_matfile(data_dir, 1 if dataset_name == 'SEGSalt' else 1601, test_size, "test")
    else:
        data_set, label_sets = batch_read_npyfile(data_dir, 1, test_size // 500, "test")
        for i in range(data_set.shape[0]):
            vm = label_sets[0][i][0]
            max_velocity, min_velocity = np.max(vm), np.min(vm)
            label_sets[0][i][0] = (vm - min_velocity) / (max_velocity - min_velocity)

    print("· Number of seismic gathers included in the testing set: {}.".format(test_size))
    print("· Dimensions of seismic data: ({},{},{},{}).".format(test_size, inchannels, data_dim[0], data_dim[1]))
    print("· Dimensions of velocity model: ({},{},{},{}).".format(test_size, classes, model_dim[0], model_dim[1]))
    print("---------------------------------")

    seis_and_vm = data_utils.TensorDataset(torch.from_numpy(data_set).float(),
                                           torch.from_numpy(label_sets[0]).float())
    seis_and_vm_loader = data_utils.DataLoader(seis_and_vm, batch_size=test_batch_size, shuffle=True)

    return seis_and_vm_loader, data_set, label_sets

    #Marmousi数据
# def load_dataset():
#     '''
#     Load the full testing data set from all .npy files (including the last one with <500 samples)
#     '''
#
#     print("---------------------------------")
#     print("· Loading the datasets...")
#     import os
#     import glob
#
#     print("✅ Debug Info:")
#     print("当前 data_dir 路径为：", data_dir)
#
#     seismic_path_check = os.path.join(data_dir, "seismic/seismic*.npy")
#     vmodel_path_check = os.path.join(data_dir, "vmodel/vmodel*.npy")
#     print("匹配 seismic 路径：", seismic_path_check)
#     print("匹配 vmodel 路径：", vmodel_path_check)
#
#     seismic_files = sorted(glob.glob(seismic_path_check))
#     vmodel_files = sorted(glob.glob(vmodel_path_check))
#
#     print("找到 seismic 文件数：", len(seismic_files))
#     print("找到 vmodel 文件数：", len(vmodel_files))
#
#     seismic_files = sorted(glob.glob(os.path.join(data_dir, "test_data/seismic/seismic*.npy")))
#     vmodel_files = sorted(glob.glob(os.path.join(data_dir, "test_data/vmodel/vmodel*.npy")))
#
#     seismic_list = []
#     vmodel_list = []
#
#     for s_file, v_file in zip(seismic_files, vmodel_files):
#         s_data = np.load(s_file)   # (N, C, H, W)
#         v_data = np.load(v_file)   # (N, 1, H, W)
#
#         # 归一化每个速度模型（逐样本处理）
#         for i in range(v_data.shape[0]):
#             vm = v_data[i, 0]
#             vmax, vmin = np.max(vm), np.min(vm)
#             if vmax > vmin:
#                 v_data[i, 0] = (vm - vmin) / (vmax - vmin)
#
#         seismic_list.append(s_data)
#         vmodel_list.append(v_data)
#
#     data_set = np.concatenate(seismic_list, axis=0)
#     label_set = np.concatenate(vmodel_list, axis=0)
#
#     print("· Total seismic gathers loaded: {}.".format(data_set.shape[0]))
#     print("· Seismic data shape: {}".format(data_set.shape))
#     print("· Velocity model shape: {}".format(label_set.shape))
#     print("---------------------------------")
#
#     seis_and_vm = data_utils.TensorDataset(torch.from_numpy(data_set).float(),
#                                            torch.from_numpy(label_set).float())
#
#     seis_and_vm_loader = data_utils.DataLoader(seis_and_vm,
#                                                batch_size=test_batch_size,
#                                                shuffle=False,  # 测试集无需 shuffle
#                                                drop_last=False)  # 保留最后不足一个 batch 的部分
#
#     return seis_and_vm_loader, data_set, [label_set]



# def batch_test(model_path, model_type = "Deeplab"):
#     '''
#     Batch testing for multiple seismic data
#
#     :param model_path:              Model path
#     :param model_type:              The main model used, this model is differentiated based on different papers.
#                                     The available key model keywords are
#                                     [DDNet70 | DDNet | InversionNet | FCNVMB| SDNet70 | SDNet]
#     :return:
#     '''
#
#     loader, seismic_gathers, velocity_models = load_dataset()
#
#     print("Loading test model:{}".format(model_path))
#     model_net, device, optimizer = determine_network(model_path, model_type=model_type)
#
#     # Switch to evaluation mode
#     model_net.eval()
#
#     mse_record = np.zeros((1, test_size), dtype=float)
#     mae_record = np.zeros((1, test_size), dtype=float)
#     uqi_record = np.zeros((1, test_size), dtype=float)
#     lpips_record = np.zeros((1, test_size), dtype=float)
#     ssim_record = np.zeros((1, test_size), dtype=float)
#     psnr_record = np.zeros((1, test_size), dtype=float)
#
#     counter = 0
#
#     lpips_object = lpips.LPIPS(net='alex', version="0.1")
#
#     cur_node_time = time.time()
#     for i, (seis_image, gt_vmodel) in enumerate(loader):
#
#         if torch.cuda.is_available():
#             seis_image = seis_image.cuda(non_blocking=True)
#             gt_vmodel = gt_vmodel.cuda(non_blocking=True)
#
#         # Prediction
#         model_net.eval()
#         if model_type in ["DeepLabV3"]:
#             [outputs, _] = model_net(seis_image, model_dim)
#         elif model_type in ["DeepLab"]:
#             outputs = model_net(seis_image, model_dim)
#         elif model_type in ["DDNet", "DDNet70"]:
#             [outputs, _] = model_net(seis_image, model_dim)
#         elif model_type in ["SDNet", "SDNet70"]:
#             outputs = model_net(seis_image, model_dim)
#         elif model_type == "InversionNet":
#             outputs = model_net(seis_image)
#         elif model_type == "FCNVMB":
#             outputs = model_net(seis_image, model_dim)
#         else:
#             print('The "model_type" parameter selected in the batch_test(...) '
#                   'is the undefined network model keyword! Please check!')
#             exit(0)
#
#         # # Both target labels and prediction tags return to "numpy"
#         pd_vmodel = outputs.cpu().detach().numpy()
#         pd_vmodel = np.where(pd_vmodel > 0.0, pd_vmodel, 0.0)   # Delete bad points
#         gt_vmodel = gt_vmodel.cpu().detach().numpy()
#
#         # Calculate MSE, MAE, UQI and LPIPS of the current batch
#         # for k in range(test_batch_size):openfwi
#         # Marmousi
#         batch_size = pd_vmodel.shape[0]  # 或 gt_vmodel.shape[0]
#         for k in range(batch_size):
#
#             pd_vmodel_of_k = pd_vmodel[k, 0, :, :]
#             gt_vmodel_of_k = gt_vmodel[k, 0, :, :]
#
#             mse_record[0, counter]   = run_mse(pd_vmodel_of_k, gt_vmodel_of_k)
#             mae_record[0, counter]   = run_mae(pd_vmodel_of_k, gt_vmodel_of_k)
#             uqi_record[0, counter]   = run_uqi(gt_vmodel_of_k, pd_vmodel_of_k)
#             lpips_record[0, counter] = run_lpips(gt_vmodel_of_k, pd_vmodel_of_k, lpips_object)
#             ssim_record[0, counter] = SSIM(gt_vmodel_of_k, pd_vmodel_of_k)
#             psnr_record[0, counter] = PSNR(gt_vmodel_of_k, pd_vmodel_of_k)
#
#             print('The %d testing MSE: %.4f\tMAE: %.4f\tUQI: %.4f\tLPIPS: %.4f\tSSIM: %.4f\tPSNR: %.4f' %
#                   (counter, mse_record[0, counter], mae_record[0, counter],
#                    uqi_record[0, counter], lpips_record[0, counter],
#                    ssim_record[0, counter], psnr_record[0, counter]))
#             # 每10个样本可视化一次
#             if counter % 5992 == 0:
#                 # 可视化地震数据
#                 pain_openfwi_seismic_data(seis_image[k, 0, :, :].cpu().numpy())  # SEG数据集的地震数据
#                 # 可视化真实的速度模型
#                 pain_openfwi_velocity_model(gt_vmodel_of_k, np.min(gt_vmodel_of_k), np.max(gt_vmodel_of_k))  # 真实速度模型
#                 # 可视化预测的速度模型
#                 pain_openfwi_velocity_model(pd_vmodel_of_k, np.min(pd_vmodel_of_k), np.max(pd_vmodel_of_k))  # 预测速度模型
#             counter = counter + 1
#
#         time_elapsed = time.time() - cur_node_time
#     print(f"Min value: {np.min(pd_vmodel)}, Max value: {np.max(pd_vmodel)}")
#     print("The average of MSE: {:.6f}".format(mse_record.mean()))
#     print("The average of MAE: {:.6f}".format(mae_record.mean()))
#     print("The average of UQI: {:.6f}".format(uqi_record.mean()))
#     print("The average of LIPIS: {:.6f}".format(lpips_record.mean()))
#     print("The average of SSIM: {:.6f}".format(ssim_record.mean()))
#     print("The average of PSNR: {:.6f}".format(psnr_record.mean()))
#     print("-----------------")
#     print("Time-consuming testing of batch samples: {:.6f}".format(time_elapsed))
#     print("Average test-consuming per sample: {:.6f}".format(time_elapsed / test_size))

# def batch_test(model_path, model_type="Deeplab", save_dir="predicted_models"):
#     '''
#     Batch testing for multiple seismic data
#
#     :param model_path:              Model path
#     :param model_type:              The main model used
#     :param save_dir:                Directory to save predicted velocity models
#     :return:
#     '''
#     import os
#     import matplotlib.pyplot as plt
#
#     # Create save directory if it doesn't exist
#     os.makedirs(save_dir, exist_ok=True)
#
#     loader, seismic_gathers, velocity_models = load_dataset()
#
#     print("Loading test model:{}".format(model_path))
#     model_net, device, optimizer = determine_network(model_path, model_type=model_type)
#
#     # Switch to evaluation mode
#     model_net.eval()
#
#     mse_record = np.zeros((1, test_size), dtype=float)
#     mae_record = np.zeros((1, test_size), dtype=float)
#     uqi_record = np.zeros((1, test_size), dtype=float)
#     lpips_record = np.zeros((1, test_size), dtype=float)
#     ssim_record = np.zeros((1, test_size), dtype=float)
#     psnr_record = np.zeros((1, test_size), dtype=float)
#
#     counter = 0
#
#     lpips_object = lpips.LPIPS(net='alex', version="0.1")
#
#     cur_node_time = time.time()
#     for i, (seis_image, gt_vmodel) in enumerate(loader):
#
#         if torch.cuda.is_available():
#             seis_image = seis_image.cuda(non_blocking=True)
#             gt_vmodel = gt_vmodel.cuda(non_blocking=True)
#
#         # Prediction
#         model_net.eval()
#         if model_type in ["DeepLabV3"]:
#             [outputs, _] = model_net(seis_image, model_dim)
#         elif model_type in ["DeepLab"]:
#             outputs = model_net(seis_image, model_dim)
#         elif model_type in ["DDNet", "DDNet70"]:
#             [outputs, _] = model_net(seis_image, model_dim)
#         elif model_type in ["SDNet", "SDNet70"]:
#             outputs = model_net(seis_image, model_dim)
#         elif model_type == "InversionNet":
#             outputs = model_net(seis_image)
#         elif model_type == "FCNVMB":
#             outputs = model_net(seis_image, model_dim)
#         else:
#             print('The "model_type" parameter selected in the batch_test(...) '
#                   'is the undefined network model keyword! Please check!')
#             exit(0)
#
#         # Both target labels and prediction tags return to "numpy"
#         pd_vmodel = outputs.cpu().detach().numpy()
#         pd_vmodel = np.where(pd_vmodel > 0.0, pd_vmodel, 0.0)  # Delete bad points
#         gt_vmodel = gt_vmodel.cpu().detach().numpy()
#
#         # Calculate MSE, MAE, UQI and LPIPS of the current batch
#         batch_size = pd_vmodel.shape[0]
#         for k in range(batch_size):
#             pd_vmodel_of_k = pd_vmodel[k, 0, :, :]
#             gt_vmodel_of_k = gt_vmodel[k, 0, :, :]
#
#             mse_record[0, counter] = run_mse(pd_vmodel_of_k, gt_vmodel_of_k)
#             mae_record[0, counter] = run_mae(pd_vmodel_of_k, gt_vmodel_of_k)
#             uqi_record[0, counter] = run_uqi(gt_vmodel_of_k, pd_vmodel_of_k)
#             lpips_record[0, counter] = run_lpips(gt_vmodel_of_k, pd_vmodel_of_k, lpips_object)
#             ssim_record[0, counter] = SSIM(gt_vmodel_of_k, pd_vmodel_of_k)
#             psnr_record[0, counter] = PSNR(gt_vmodel_of_k, pd_vmodel_of_k)
#
#             print('The %d testing MSE: %.4f\tMAE: %.4f\tUQI: %.4f\tLPIPS: %.4f\tSSIM: %.4f\tPSNR: %.4f' %
#                   (counter, mse_record[0, counter], mae_record[0, counter],
#                    uqi_record[0, counter], lpips_record[0, counter],
#                    ssim_record[0, counter], psnr_record[0, counter]))
#
#
#             # 每10个样本可视化一次
#             if counter % 5992 == 0:
#                 # 可视化地震数据
#                 pain_openfwi_seismic_data(seis_image[k, 0, :, :].cpu().numpy())  # SEG数据集的地震数据
#                 # 可视化真实的速度模型
#                 pain_openfwi_velocity_model(gt_vmodel_of_k, np.min(gt_vmodel_of_k), np.max(gt_vmodel_of_k))  # 真实速度模型
#                 # 可视化预测的速度模型
#                 pain_openfwi_velocity_model(pd_vmodel_of_k, np.min(pd_vmodel_of_k), np.max(pd_vmodel_of_k))  # 预测速度模型
#             counter = counter + 1
#
#         time_elapsed = time.time() - cur_node_time
#
#     print(f"Min value: {np.min(pd_vmodel)}, Max value: {np.max(pd_vmodel)}")
#     print("The average of MSE: {:.6f}".format(mse_record.mean()))
#     print("The average of MAE: {:.6f}".format(mae_record.mean()))
#     print("The average of UQI: {:.6f}".format(uqi_record.mean()))
#     print("The average of LIPIS: {:.6f}".format(lpips_record.mean()))
#     print("The average of SSIM: {:.6f}".format(ssim_record.mean()))
#     print("The average of PSNR: {:.6f}".format(psnr_record.mean()))
#     print("-----------------")
#     print("Time-consuming testing of batch samples: {:.6f}".format(time_elapsed))
#     print("Average test-consuming per sample: {:.6f}".format(time_elapsed / test_size))
import scipy.io



def batch_test(model_path, model_type="DeepLab", save_dir="predicted_models"):
    import os
    import numpy as np
    import time
    import lpips
    import torch
    import scipy.io

    # 创建保存目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)

    loader, seismic_gathers, velocity_models = load_dataset()

    # ⭐ 自动确定样本总数
    num_samples = len(loader.dataset)
    print("Loading test model:{}".format(model_path))
    model_net, device, optimizer = determine_network(model_path, model_type=model_type)

    # 切换为评估模式
    model_net.eval()

    all_predictions = []

    # ⭐ 根据样本总数来初始化数组，而不是 test_size
    mse_record = np.zeros((1, num_samples), dtype=float)
    mae_record = np.zeros((1, num_samples), dtype=float)
    uqi_record = np.zeros((1, num_samples), dtype=float)
    lpips_record = np.zeros((1, num_samples), dtype=float)
    ssim_record = np.zeros((1, num_samples), dtype=float)
    psnr_record = np.zeros((1, num_samples), dtype=float)
    local_mse_record = np.zeros((1, num_samples), dtype=float)
    local_mae_record = np.zeros((1, num_samples), dtype=float)

    counter = 0
    lpips_object = lpips.LPIPS(net='alex', version="0.1")

    cur_node_time = time.time()

    for i, (seis_image, gt_vmodel) in enumerate(loader):


        if torch.cuda.is_available():
            seis_image = seis_image.cuda(non_blocking=True)
            gt_vmodel = gt_vmodel.cuda(non_blocking=True)

        # 预测
        model_net.eval()
        if model_type in ["DeepLabV3"]:
            [outputs, _] = model_net(seis_image, model_dim)
        elif model_type in ["DeepL","DeepLab","F3"]:
            outputs= model_net(seis_image, model_dim)
        elif model_type == "DOF":
            B, C, H, W = seis_image.shape  # 获取 batch 大小
            num_shots = 5  # OpenFWI 每个样本 5 个震源
            # 均匀分布 x 坐标
            x_coords = torch.linspace(0, W - 1, num_shots, device=seis_image.device)
            # z 坐标固定为 0（地表）
            z_coords = torch.zeros(num_shots, device=seis_image.device)
            # 组合成 [num_shots, 2]
            single_cond = torch.stack([x_coords, z_coords], dim=1)
            # 扩展成 batch 维度 [B, num_shots, 2]
            cond_tensor = single_cond.unsqueeze(0).repeat(B, 1, 1)

            # forward
            outputs = model_net(seis_image, cond_tensor, model_dim)
        elif model_type in ["DDNet", "DDNet70"]:
            [outputs, _] = model_net(seis_image, model_dim)
        elif model_type in ["SDNet", "SDNet70"]:
            outputs = model_net(seis_image, model_dim)
        elif model_type == "InversionNet":
            outputs = model_net(seis_image)
        elif model_type == "FCNVMB":
            outputs = model_net(seis_image, model_dim)
        elif model_type in ["ABA", "ABA-FWI", "ABA_FWI"]:
            outputs = model_net(seis_image)
        else:
            print('The "model_type" parameter selected in the batch_test(...) is undefined! Please check!')
            exit(0)

        pd_vmodel = outputs.cpu().detach().numpy()
        pd_vmodel = np.where(pd_vmodel > 0.0, pd_vmodel, 0.0)

        all_predictions.append(pd_vmodel)

        gt_vmodel = gt_vmodel.cpu().detach().numpy()

        batch_size = pd_vmodel.shape[0]

        for k in range(batch_size):
            pd_vmodel_of_k = pd_vmodel[k, 0, :, :]
            gt_vmodel_of_k = gt_vmodel[k, 0, :, :]

            mse_record[0, counter] = run_mse(pd_vmodel_of_k, gt_vmodel_of_k)
            mae_record[0, counter] = run_mae(pd_vmodel_of_k, gt_vmodel_of_k)
            uqi_record[0, counter] = run_uqi(gt_vmodel_of_k, pd_vmodel_of_k)
            lpips_record[0, counter] = run_lpips(gt_vmodel_of_k, pd_vmodel_of_k, lpips_object)
            ssim_record[0, counter] = SSIM(gt_vmodel_of_k, pd_vmodel_of_k)
            psnr_record[0, counter] = PSNR(gt_vmodel_of_k, pd_vmodel_of_k)
            edge = extract_contours(gt_vmodel_of_k)
            local_mse_record[0, counter] = local_MSE(pd_vmodel_of_k, gt_vmodel_of_k, edge)
            local_mae_record[0, counter] = local_MAE(pd_vmodel_of_k, gt_vmodel_of_k, edge)

            print('The %d testing MSE: %.4f\tMAE: %.4f\tUQI: %.4f\tLPIPS: %.4f\tSSIM: %.4f\tPSNR: %.4f\tLocal-MSE: %.4f\tLocal-MAE: %.4f' %
                  (counter, mse_record[0, counter], mae_record[0, counter],
                   uqi_record[0, counter], lpips_record[0, counter],
                   ssim_record[0, counter], psnr_record[0, counter],local_mse_record[0, counter], local_mae_record[0, counter]))

            if counter % 5992 == 0:
                if dataset_name in ['SEGSalt', 'SEGSimulation']:
                    pain_seg_seismic_data(seis_image[k, 14, :, :].cpu().numpy())
                    pain_seg_velocity_model(gt_vmodel_of_k, np.min(gt_vmodel_of_k), np.max(gt_vmodel_of_k))
                    pain_seg_velocity_model(pd_vmodel_of_k, np.min(pd_vmodel_of_k), np.max(pd_vmodel_of_k))
                else:
                    pain_openfwi_seismic_data1(seis_image[k, 2, :, :].cpu().numpy())
                    pain_openfwi_velocity_model1(gt_vmodel_of_k, np.min(gt_vmodel_of_k), np.max(gt_vmodel_of_k))
                    pain_openfwi_velocity_model1(pd_vmodel_of_k, np.min(pd_vmodel_of_k), np.max(pd_vmodel_of_k))

            counter += 1

    time_elapsed = time.time() - cur_node_time

    all_predictions = np.concatenate(all_predictions, axis=0)

    # 保存为 .mat 文件，变量名为 'Prediction'
    save_path = os.path.join(save_dir, "predicted_velocity_models.mat")
    scipy.io.savemat(save_path, {'Prediction': all_predictions})
    print(f"Predicted velocity models saved to: {save_path}")

    print(f"Min value: {np.min(all_predictions)}, Max value: {np.max(all_predictions)}")
    print("The average of MSE: {:.6f}".format(mse_record.mean()))
    print("The average of MAE: {:.6f}".format(mae_record.mean()))
    print("The average of UQI: {:.6f}".format(uqi_record.mean()))
    print("The average of LIPIS: {:.6f}".format(lpips_record.mean()))
    print("The average of SSIM: {:.6f}".format(ssim_record.mean()))
    print("The average of PSNR: {:.6f}".format(psnr_record.mean()))
    print("The average of Local MSE: {:.6f}".format(local_mse_record.mean()))
    print("The average of Local MAE: {:.6f}".format(local_mae_record.mean()))
    print("-----------------")
    print("Time-consuming testing of batch samples: {:.6f}".format(time_elapsed))
    print("Average test-consuming per sample: {:.6f}".format(time_elapsed / test_size))



def single_test(model_path, select_id, train_or_test = "test", model_type = "DeepLab"):
    '''
    Batch testing for single seismic data

    :param model_path:              Model path
    :param select_id:               The ID of the selected data. if it is openfwi, here is a pair,
                                    e.g. [11, 100], otherwise it is just a single number, e.g. 56.
    :param train_or_test:           Whether the data set belongs to the training set or the testing set
    :param model_type:              The main model used, this model is differentiated based on different papers.
                                    The available key model keywords are
                                    [DDNet70 | DDNet | InversionNet | FCNVMB| SDNet70 | SDNet]
    :return:
    '''

    print("Loading test model:{}".format(model_path))
    model_net, device, optimizer = determine_network(model_path, model_type=model_type)

    if dataset_name in ['SEGSalt', 'SEGSimulation']:
        seismic_data, velocity_model, _ = single_read_matfile(data_dir, data_dim, model_dim, select_id, train_or_test = train_or_test)
        max_velocity, min_velocity = np.max(velocity_model), np.min(velocity_model)
    else:
        seismic_data, velocity_model, _ = single_read_npyfile(data_dir, select_id, train_or_test = train_or_test)
        max_velocity, min_velocity = np.max(velocity_model), np.min(velocity_model)
        velocity_model = (velocity_model - np.min(velocity_model)) / (np.max(velocity_model) - np.min(velocity_model))

    lpips_object = lpips.LPIPS(net='alex', version="0.1")


    # Convert numpy to tensor and load it to GPU
    seismic_data_tensor = torch.from_numpy(np.array([seismic_data])).float()
    if torch.cuda.is_available():
        seismic_data_tensor = seismic_data_tensor.cuda(non_blocking=True)

    # Prediction
    model_net.eval()
    cur_node_time = time.time()
    if model_type in ["DDNet", "DDNet70"]:
        [predicted_vmod_tensor, _] = model_net(seismic_data_tensor, model_dim)
    elif model_type in ["SDNet", "SDNet70"]:
        predicted_vmod_tensor = model_net(seismic_data_tensor, model_dim)
    elif model_type == "InversionNet":
        predicted_vmod_tensor = model_net(seismic_data_tensor)
    elif model_type == "FCNVMB":
        predicted_vmod_tensor = model_net(seismic_data_tensor, model_dim)
    elif model_type in ["ABA", "ABA-FWI", "ABA_FWI"]:
        predicted_vmod_tensor = model_net(seismic_data_tensor)
    elif model_type in ["DeepL","DeepLab","F3"]:
        predicted_vmod_tensor = model_net(seismic_data_tensor, model_dim)  # DeepLab只返回一个输出
    else:
        print('The "model_type" parameter selected in the single_test(...) '
              'is the undefined network model keyword! Please check!')
        exit(0)
    time_elapsed = time.time() - cur_node_time

    predicted_vmod = predicted_vmod_tensor.cpu().detach().numpy()[0][0]     # (1, 1, X, X)
    predicted_vmod = np.where(predicted_vmod > 0.0, predicted_vmod, 0.0)    # Delete bad points

    # -----------------------------
    # 保存预测速度模型和Ground Truth
    # -----------------------------
    test_result_dir = './VelocityCFA/'
    os.makedirs(test_result_dir, exist_ok=True)

    # 保存预测结果
    np.save(os.path.join(test_result_dir, f'{model_type}_pred_{select_id}.npy'), predicted_vmod)
    #保存Ground Truth
    if dataset_name in ['SEGSalt', 'SEGSimulation']:
        np.save(os.path.join(test_result_dir, f'GT_{select_id}.npy'), velocity_model)
    else:
        np.save(os.path.join(test_result_dir, f'GT_{select_id}.npy'),
                min_velocity + velocity_model * (max_velocity - min_velocity))




    mse   = run_mse(predicted_vmod, velocity_model)
    mae   = run_mae(predicted_vmod, velocity_model)
    uqi   = run_uqi(velocity_model, predicted_vmod)
    lpi = run_lpips(velocity_model, predicted_vmod, lpips_object)
    psnr = PSNR(velocity_model, predicted_vmod)
    ssim = SSIM(velocity_model, predicted_vmod)

    print('MSE: %.6f\nMAE: %.6f\nUQI: %.6f\nLPIPS: %.6f\nSSIM: %.6f\nPSNR: %.6f' % (mse, mae, uqi, lpi, ssim, psnr))
    print("-----------------")
    print("Time-consuming testing of a sample: {:.6f}".format(time_elapsed))

    # Show
    if dataset_name in ['SEGSalt', 'SEGSimulation']:
        pain_seg_seismic_data(seismic_data[15])
        pain_seg_velocity_model(velocity_model, min_velocity, max_velocity)
        pain_seg_velocity_model(predicted_vmod, min_velocity, max_velocity)
        # -----------------------------
        # 可视化速度模型的边界
        # -----------------------------
        # gt_edges = extract_contours(velocity_model)
        # pred_edges = extract_contours(predicted_vmod)
        #
        # plt.figure(figsize=(10, 4))
        # plt.subplot(1, 2, 1)
        # plt.title("GT Velocity Contours")
        # plt.imshow(gt_edges, cmap='gray', aspect='auto')
        #
        # plt.subplot(1, 2, 2)
        # plt.title("Predicted Velocity Contours")
        # plt.imshow(pred_edges, cmap='gray', aspect='auto')
        # plt.show()

    else:
        residual = plot_velocity_residual_only(velocity_model, predicted_vmod)
        pain_openfwi_seismic_data1(seismic_data[2])
        minV = np.min(min_velocity + velocity_model * (max_velocity - min_velocity))
        maxV = np.max(min_velocity + velocity_model * (max_velocity - min_velocity))
        pain_openfwi_velocity_model1(min_velocity + velocity_model * (max_velocity - min_velocity), minV, maxV)
        pain_openfwi_velocity_model1(min_velocity + predicted_vmod * (max_velocity - min_velocity), minV, maxV)
        # 假设 velocity_model 和 predicted_vmod 已经是 70x70 的二维 numpy 数组
        gt_edges = extract_contours(velocity_model)
        pred_edges = extract_contours(predicted_vmod)

        # 荧光黄色边缘
        edge_color = '#FFFF33'

        # 深蓝偏紫色背景
        bg_color = np.zeros((70, 70, 3))
        bg_color[..., 0] = 0.2  # R
        bg_color[..., 1] = 0.0  # G
        bg_color[..., 2] = 0.3  # B，调高一点偏紫

        # # 可视化真实速度模型边界
        # plt.figure(figsize=(5, 5))
        # plt.title("GT Velocity Contours", fontsize=14, fontweight='bold', color='white')
        # plt.imshow(bg_color, aspect='auto')  # 深蓝色背景
        # plt.contour(gt_edges, levels=[0.5], colors=edge_color, linewidths=2)
        # plt.axis('on')  # 显示坐标
        # plt.show()
        #
        # # 可视化预测速度模型边界
        # plt.figure(figsize=(5, 5))
        # plt.title("Predicted Velocity Contours", fontsize=14, fontweight='bold', color='white')
        # plt.imshow(bg_color, aspect='auto')  # 深蓝色背景
        # plt.contour(pred_edges, levels=[0.5], colors=edge_color, linewidths=2)
        # plt.axis('on')  # 显示坐标
        # plt.show()







if __name__ == "__main__":
    batch_of_single =1
    # |DDNet|DDNet70|InversionNet|FCNVMB|SDNet|SDNet70|DeepLab|ABA_FWI
    model_type = ("F3")

    if batch_of_single == 1:
        # Batch test #
        batch_test("", model_type=model_type)

    else:
        # Single test #
        if dataset_name in ["SEGSalt", "SEGSimulation"]:
            # 1~10      :SEGSalt
            # 1601~1700 :SEGSimulation
            select_id = 1
        else:
            # [1~2, 0~499]
            select_id = [1,1]
        single_test("", select_id=select_id, model_type=model_type)

