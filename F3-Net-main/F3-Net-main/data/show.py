# -*- coding: utf-8 -*-
"""
Created on 2023/10/20 9:05

@author: XUQIONG  (xuqiong@swpu.edu.cn)

"""
import matplotlib.pylab as plt
import matplotlib as mpl

import matplotlib.patches as patches
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
mpl.use('TkAgg')

import numpy as np
import cv2


font21 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 21,
}

font18 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 18,
}


def pain_seg_seismic_data(para_seismic_data):
    """
    Plotting seismic data images of SEG salt datasets

    :param para_seismic_data:  Seismic data (400 x 301) (numpy)
    :param is_colorbar: Whether to add a color bar (1 means add, 0 is the default, means don't add)
    """
    fig, ax = plt.subplots(figsize=(6.2, 8), dpi = 120)

    im = ax.imshow(para_seismic_data, extent=[0, 300, 400, 0], cmap=plt.cm.seismic, vmin=-0.4, vmax=0.44)

    ax.set_xlabel('Position (km)', font21)
    ax.set_ylabel('Time (s)', font21)

    ax.set_xticks(np.linspace(0, 300, 5))
    ax.set_yticks(np.linspace(0, 400, 5))
    ax.set_xticklabels(labels = [0,0.75,1.5,2.25,3.0], size=21)
    ax.set_yticklabels(labels = [0.0,0.50,1.00,1.50,2.00], size=21)

    plt.rcParams['font.size'] = 14      # Set colorbar font size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="3%", pad=0.32)
    plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal')
    plt.subplots_adjust(bottom=0.08, top=0.98, left=0.11, right=0.99)

    plt.show()


def pain_openfwi_seismic_data1(para_seismic_data):
    """
    Plotting seismic data images of openfwi dataset

    :param para_seismic_data:   Seismic data (1000 x 70) (numpy)
    """
    data = cv2.resize(para_seismic_data, dsize=(400, 301), interpolation=cv2.INTER_CUBIC)
    fig, ax = plt.subplots(figsize=(6.1, 8), dpi = 120)
    im = ax.imshow(data, extent=[0, 0.7, 1.0, 0], cmap=plt.cm.seismic, vmin=-18, vmax=19)

    ax.set_xlabel('Position (km)', font21)
    ax.set_ylabel('Time (s)', font21)

    ax.set_xticks(np.linspace(0, 0.7, 5))
    ax.set_yticks(np.linspace(0, 1.0, 5))
    ax.set_xticklabels(labels=[0, 0.17, 0.35, 0.52, 0.7], size=21)
    ax.set_yticklabels(labels=[0, 0.25, 0.5, 0.75, 1.0], size=21)

    plt.rcParams['font.size'] = 14      # Set colorbar font size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="3%", pad=0.3)
    plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal')
    plt.subplots_adjust(bottom=0.08, top=0.98, left=0.11, right=0.99)

    plt.show()
    plt.close()



def pain_openfwi_velocity_model1(para_velocity_model, min_velocity, max_velocity, is_colorbar = 1):
    '''
    Plotting seismic data images of openfwi dataset

    :param para_velocity_model: Velocity model (70 x 70) (numpy)
    :param min_velocity:        Upper limit of velocity in the velocity model
    :param max_velocity:        Lower limit of velocity in the velocity model
    :param is_colorbar:         Whether to add a color bar (1 means add, 0 is the default, means don't add)
    :return:
    '''

    if is_colorbar == 0:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    else:
        fig, ax = plt.subplots(figsize=(5.8, 6), dpi=150)

    im = ax.imshow(para_velocity_model, extent=[0, 0.7, 0.7, 0], vmin=min_velocity, vmax=max_velocity)

    ax.set_xlabel('Position (km)', font18)
    ax.set_ylabel('Depth (km)', font18)
    ax.set_xticks(np.linspace(0, 0.7, 8))
    ax.set_yticks(np.linspace(0, 0.7, 8))
    ax.set_xticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=18)
    ax.set_yticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=18)

    if is_colorbar == 0:
        plt.subplots_adjust(bottom=0.11, top=0.95, left=0.11, right=0.95)
    else:
        plt.rcParams['font.size'] = 14      # Set colorbar font size
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="3%", pad=0.35)
        plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal',
                     ticks=np.linspace(min_velocity, max_velocity, 7), format = mpl.ticker.StrMethodFormatter('{x:.0f}'))
        plt.subplots_adjust(bottom=0.10, top=0.95, left=0.13, right=0.95)

    plt.show()





def pain_marmousi_velocity_model(para_velocity_model, min_velocity, max_velocity, is_colorbar = 1):
    '''
    Plotting seismic data images of openfwi dataset

    :param para_velocity_model: Velocity model (70 x 70) (numpy)
    :param min_velocity:        Upper limit of velocity in the velocity model
    :param max_velocity:        Lower limit of velocity in the velocity model
    :param is_colorbar:         Whether to add a color bar (1 means add, 0 is the default, means don't add)
    :return:
    '''

    if is_colorbar == 0:
        fig, ax = plt.subplots(figsize=(10, 3.2), dpi=150)
    else:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=150)

    im = ax.imshow(para_velocity_model, extent=[0, 17, 3.5, 0], vmin=min_velocity, vmax=max_velocity, cmap='jet')

    ax.set_xlabel('Position (km)', font18)
    ax.set_ylabel('Depth (km)', font18)
    ax.set_xticks(range(0, 18, 2))
    ax.set_xticks([17], minor=True)
    ax.set_yticks(np.linspace(0, 3.5, 8))
    ax.set_xticklabels(labels=[' ', 2, 4, 6, 8, 10, 12, 14, 16], size=12)
    ax.set_yticklabels(labels=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5], size=12)
    x_start, x_end = 7350, 8050
    y_start, y_end = 420, 280
    rect = patches.Rectangle((x_start, y_start), x_end-x_start, y_end-y_start, linewidth=10, edgecolor='r', facecolor='none')

    # 将矩形补丁添加到坐标轴上
    ax.add_patch(rect)


    if is_colorbar == 0:
        plt.subplots_adjust(bottom=0.11, top=0.95, left=0.11, right=0.95)
    else:
        plt.rcParams['font.size'] = 14      # Set colorbar font size
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="3%", pad=0.35)
        plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal',
                     ticks=np.linspace(min_velocity, max_velocity, 7), format = mpl.ticker.StrMethodFormatter('{x:.0f}'))
        plt.subplots_adjust(bottom=0.10, top=0.95, left=0.13, right=0.95)

    plt.show()

def pain_openfwi_seismic_data1(para_seismic_data, is_colorbar = 1):
    '''
    Plotting seismic data images of openfwi dataset

    :param para_seismic_data:   Seismic data (1000 x 70) (numpy)
    :param is_colorbar:         Whether to add a color bar (1 means add, 0 is the default, means don't add)
    '''

    # The size of 1000 x 70 is not easy to display, we compressed it to a similar size of 400 x 301 as the SEG dataset.
    data = cv2.resize(para_seismic_data, dsize=(400, 301), interpolation=cv2.INTER_CUBIC)   #

    if is_colorbar == 0:
        fig, ax = plt.subplots(figsize=(6.5, 8), dpi = 120)
    else:
        fig, ax = plt.subplots(figsize=(6.2, 8), dpi = 120)

    im = ax.imshow(data, extent=[0, 0.7, 1.0, 0], cmap=plt.cm.seismic, vmin=-18, vmax=19)
    ax.set_xlabel('Position (km)', font21)
    ax.set_ylabel('Time (s)', font21)
    ax.set_xticks(np.linspace(0, 0.7, 5))
    ax.set_yticks(np.linspace(0, 1.0, 5))
    ax.set_xticklabels(labels=[0, 0.17, 0.35, 0.52, 0.7], size=21)
    ax.set_yticklabels(labels=[0, 0.25, 0.5, 0.75, 1.0], size=21)

    if is_colorbar == 0:
        plt.subplots_adjust(bottom=0.11, top=0.95, left=0.11, right=0.99)
    else:
        plt.rcParams['font.size'] = 14      # Set colorbar font size
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="3%", pad=0.3)
        plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal')

        plt.subplots_adjust(bottom=0.08, top=0.98, left=0.11, right=0.99)

    plt.show()


# def plot_velocity_image_transfer(num, GISP, ABA, DDNet70, VelocityGAN, InversionNet, target,
#                                  test_result_dir, vmin, vmax, column_index=35):
#     """
#     Plot vertical velocity curves for GISP, ABA-FWI, DDNet70, VelocityGAN, InversionNet, and Ground Truth.
#     """
#
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import os
#
#     fig = plt.figure(figsize=(9, 7))
#
#     # 设置字体
#     plt.rcParams['font.family'] = 'Times New Roman'
#
#     pixel_values_GISP, pixel_values_ABA, pixel_values_DDNet70, pixel_values_VG, pixel_values_IN, pixel_values_GT = [], [], [], [], [], []
#
#     for y in range(target.shape[0]):
#         pixel_values_GISP.append(GISP[y, column_index])
#         pixel_values_ABA.append(ABA[y, column_index])
#         pixel_values_DDNet70.append(DDNet70[y, column_index])
#         pixel_values_VG.append(VelocityGAN[y, column_index])
#         pixel_values_IN.append(InversionNet[y, column_index])
#         pixel_values_GT.append(target[y, column_index])
#
#     # 绘制曲线
#     plt.plot(pixel_values_GT, color='orange', linewidth=2, label='Ground Truth')
#     plt.plot(pixel_values_IN, color='cyan', linewidth=2, label='InversionNet')
#     plt.plot(pixel_values_VG, color='purple', linewidth=2, label='VelocityGAN')
#     plt.plot(pixel_values_DDNet70, color='green', linewidth=2, label='DDNet70')
#     plt.plot(pixel_values_ABA, color='blue', linewidth=2, label='ABA-FWI')
#     plt.plot(pixel_values_GISP, color='red', linewidth=2, label='FWI-GISP')
#
#     # 图例和坐标
#     plt.legend(fontsize=14)
#     plt.xlabel('Depth (m)', fontsize=14)
#     plt.ylabel('Velocity (m/s)', fontsize=14)
#     plt.xticks(range(0, 80, 10), fontsize=12)
#
#     ticks = [np.min(vmin), 0.2 * (vmax - vmin) + vmin, 0.4 * (vmax - vmin) + vmin,
#              0.6 * (vmax - vmin) + vmin, 0.8 * (vmax - vmin) + vmin, np.max(vmax)]
#     tick_labels = [int(tick) for tick in ticks]
#     plt.yticks(ticks, tick_labels, fontsize=12)
#
#     plt.subplots_adjust(bottom=0.10, top=0.92, left=0.12, right=0.95)
#     plt.savefig(os.path.join(test_result_dir, f'TransferLearning_Velocity_{num}.png'), dpi=150)
#     plt.close(fig)
def plot_velocity_image_transfern(num, GISP, ABA, DDNet70, VelocityGAN, InversionNet, target,
                                 test_result_dir, vmin, vmax, column_index=35):
    """
    Plot vertical velocity curves for GISP, ABA-FWI, DDNet70, VelocityGAN, InversionNet, and Ground Truth.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # === 添加详细的调试信息 ===
    print(f"\n=== 绘图函数内部调试 ===")
    print(f"传入的 column_index: {column_index}")
    print(f"target形状: {target.shape}")
    print(f"GISP形状: {GISP.shape}")

    # 检查实际提取的数据
    test_depth_points = [20, 35, 50]
    for depth in test_depth_points:
        print(
            f"深度{depth},列{column_index}: GT={target[depth, column_index]:.1f}, GISP={GISP[depth, column_index]:.1f}")

    fig = plt.figure(figsize=(9, 7))

    # 设置字体
    plt.rcParams['font.family'] = 'Times New Roman'

    pixel_values_GISP, pixel_values_ABA, pixel_values_DDNet70, pixel_values_VG, pixel_values_IN, pixel_values_GT = [], [], [], [], [], []

    # === 检查循环提取过程 ===
    print(f"开始提取列{column_index}的数据...")
    for y in range(target.shape[0]):
        pixel_values_GISP.append(GISP[y, column_index])
        pixel_values_ABA.append(ABA[y, column_index])
        pixel_values_DDNet70.append(DDNet70[y, column_index])
        pixel_values_VG.append(VelocityGAN[y, column_index])
        pixel_values_IN.append(InversionNet[y, column_index])
        pixel_values_GT.append(target[y, column_index])

        # 打印前几个点的提取值用于验证
        if y < 3:
            print(f"  深度{y}: GT={target[y, column_index]:.1f}, GISP={GISP[y, column_index]:.1f}")

    # === 检查最终提取的数据 ===
    print(f"提取完成，数据点数量: {len(pixel_values_GT)}")
    print(f"GT曲线范围: {min(pixel_values_GT):.1f} ~ {max(pixel_values_GT):.1f}")
    print(f"GISP曲线范围: {min(pixel_values_GISP):.1f} ~ {max(pixel_values_GISP):.1f}")

    # 绘制曲线
    plt.plot(pixel_values_GT, color='orange', linewidth=2, label='Ground Truth')
    plt.plot(pixel_values_IN, color='cyan', linewidth=2, label='InversionNet')
    plt.plot(pixel_values_VG, color='purple', linewidth=2, label='VelocityGAN')
    plt.plot(pixel_values_DDNet70, color='green', linewidth=2, label='DDNet70')
    plt.plot(pixel_values_ABA, color='blue', linewidth=2, label='ABA-FWI')
    plt.plot(pixel_values_GISP, color='red', linewidth=2, label='F$^3$-net')

    # 图例和坐标
    plt.legend(fontsize=14)
    plt.xlabel('Depth (m)', fontsize=14)
    plt.ylabel('Velocity (m/s)', fontsize=14)
    plt.xticks(range(0, 80, 10), fontsize=12)

    ticks = [np.min(vmin), 0.2 * (vmax - vmin) + vmin, 0.4 * (vmax - vmin) + vmin,
             0.6 * (vmax - vmin) + vmin, 0.8 * (vmax - vmin) + vmin, np.max(vmax)]
    tick_labels = [int(tick) for tick in ticks]
    plt.yticks(ticks, tick_labels, fontsize=12)

    plt.subplots_adjust(bottom=0.10, top=0.92, left=0.12, right=0.95)
    plt.savefig(os.path.join(test_result_dir, f'TransferLearning_Velocity_{num}.png'), dpi=150)
    plt.close(fig)

    print("=== 绘图函数执行完成 ===\n")

def plot_SEG_image_transfern(num, GISP, DDNet, FCNVMB, GT,
                                 test_result_dir, vmin, vmax, column_index=35):
    """
    Plot vertical velocity curves for available models and Ground Truth.
    Only plots: GISP, DDNet, FCNVMB, GT.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # 调试信息
    print(f"\n=== 绘图函数内部调试 ===")
    print(f"传入的 column_index: {column_index}")
    print(f"GT形状: {GT.shape}")
    print(f"GISP形状: {GISP.shape}")

    # 检查实际提取的数据
    test_depth_points = [20, 35, 50]
    for depth in test_depth_points:
        print(f"深度{depth},列{column_index}: GT={GT[depth, column_index]:.1f}, "
              f"GISP={GISP[depth, column_index]:.1f}, "
              f"DDNet={DDNet[depth, column_index]:.1f}, "
              f"FCNVMB={FCNVMB[depth, column_index]:.1f}")

    fig = plt.figure(figsize=(9, 7))
    plt.rcParams['font.family'] = 'Times New Roman'

    # 提取指定列的数据
    pixel_values_GISP = GISP[:, column_index]
    pixel_values_DDNet = DDNet[:, column_index]
    pixel_values_FCNVMB = FCNVMB[:, column_index]
    pixel_values_GT = GT[:, column_index]

    # 打印前几个点检查
    for y in range(3):
        print(f"深度{y}: GT={pixel_values_GT[y]:.1f}, "
              f"GISP={pixel_values_GISP[y]:.1f}, "
              f"DDNet={pixel_values_DDNet[y]:.1f}, "
              f"FCNVMB={pixel_values_FCNVMB[y]:.1f}")

    print(f"数据点数量: {len(pixel_values_GT)}")
    print(f"GT曲线范围: {pixel_values_GT.min():.1f} ~ {pixel_values_GT.max():.1f}")
    print(f"GISP曲线范围: {pixel_values_GISP.min():.1f} ~ {pixel_values_GISP.max():.1f}")

    # 绘制曲线
    plt.plot(pixel_values_GT, color='orange', linewidth=2, label='Ground Truth')
    plt.plot(pixel_values_DDNet, color='green', linewidth=2, label='DDNet')
    plt.plot(pixel_values_FCNVMB, color='purple', linewidth=2, label='FCNVMB')
    plt.plot(pixel_values_GISP, color='red', linewidth=2, label='FWI-GISP+')

    # 图例和坐标
    plt.legend(fontsize=14)
    plt.xlabel('Depth (m)', fontsize=14)
    plt.ylabel('Velocity (m/s)', fontsize=14)
    plt.xticks(range(0, 80, 10), fontsize=12)

    # 自定义y轴
    ticks = [vmin, 0.2*(vmax-vmin)+vmin, 0.4*(vmax-vmin)+vmin, 0.6*(vmax-vmin)+vmin, 0.8*(vmax-vmin)+vmin, vmax]
    tick_labels = [int(tick) for tick in ticks]
    plt.yticks(ticks, tick_labels, fontsize=12)

    plt.subplots_adjust(bottom=0.10, top=0.92, left=0.12, right=0.95)
    plt.savefig(os.path.join(test_result_dir, f'TransferLearning_Velocity_{num}.png'), dpi=150)
    plt.close(fig)

    print("=== 绘图函数执行完成 ===\n")


def pain_seg_velocity_model(para_velocity_model):
    """
    :param para_velocity_model: Velocity model (200 x 301) (numpy)
    :param min_velocity: Upper limit of velocity in the velocity model
    :param max_velocity: Lower limit of velocity in the velocity model
    :return:
    """
    fig, ax = plt.subplots(figsize=(5.8, 4.3), dpi=150)
    im = ax.imshow(para_velocity_model, extent=[0, 3, 2, 0])

    ax.set_xlabel('Position (km)', font18)
    ax.set_ylabel('Depth (km)', font18)
    ax.tick_params(labelsize=14)

    plt.rcParams['font.size'] = 14  # Set colorbar font size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="3%", pad=0.32)
    plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal')
    plt.subplots_adjust(bottom=0.12, top=0.95, left=0.11, right=0.99)

    plt.show()


def plot_velocity_residual_only(gt_data, pred_data):
    """
    只显示速度模型残差图

    :param gt_data: 真实速度模型 (70 x 70) (numpy)
    :param pred_data: 预测速度模型 (70 x 70) (numpy)
    :return: 残差矩阵
    """
    # 计算残差
    residual = gt_data - pred_data

    # 创建图形 - 现在只有一个子图
    fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=150)

    # 残差热图
    im = ax.imshow(residual, cmap='viridis',
                   extent=[0, 0.7, 0.7, 0], aspect='auto')
    ax.set_xlabel('Position (km)', fontsize=14)
    ax.set_ylabel('Depth (km)', fontsize=14)
    ax.tick_params(labelsize=12)

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Residual (m/s)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.show()

    return residual


def PlotComparison_openfwi_seismic_data(para_raw_data, para_genenrate_data):
    """
        Compare raw seismic data images and genenrated ones of openfwi dataset

        :param para_raw_data: 15 Hz Velocity model (1000 x 70) (numpy)
        :param para_genenrate_data: different observation systems, different main frequencies Velocity model (1000 x 70) (numpy)

        :return:
    """

    data = cv2.resize(para_raw_data, dsize=(400, 301), interpolation=cv2.INTER_CUBIC)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 16), dpi=120)
    im = ax1.imshow(data, extent=[0, 0.7, 1.0, 0], cmap=plt.cm.seismic, vmin=-18, vmax=19)

    ax1.set_xlabel('Position (km)', font21)
    ax1.set_ylabel('Time (s)', font21)

    ax1.set_xticks(np.linspace(0, 0.7, 5))
    ax1.set_yticks(np.linspace(0, 1.0, 5))
    ax1.set_xticklabels(labels=[0, 0.17, 0.35, 0.52, 0.7], size=21)
    ax1.set_yticklabels(labels=[0, 0.25, 0.5, 0.75, 1.0], size=21)

    plt.rcParams['font.size'] = 14  # Set colorbar font size
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("top", size="3%", pad=0.3)
    plt.colorbar(im, ax=ax1, cax=cax1, orientation='horizontal')
    plt.subplots_adjust(bottom=0.08, top=0.98, left=0.11, right=0.99)

    generate_data = cv2.resize(para_genenrate_data, dsize=(400, 301), interpolation=cv2.INTER_CUBIC)
    im = ax2.imshow(generate_data, extent=[0, 0.7, 1.0, 0], cmap=plt.cm.seismic, vmin=-18, vmax=19)

    ax2.set_xlabel('Position (km)', font21)
    ax2.set_ylabel('Time (s)', font21)

    ax2.set_xticks(np.linspace(0, 0.7, 5))
    ax2.set_yticks(np.linspace(0, 1.0, 5))
    ax2.set_xticklabels(labels=[0, 0.17, 0.35, 0.52, 0.7], size=21)
    ax2.set_yticklabels(labels=[0, 0.25, 0.5, 0.75, 1.0], size=21)

    plt.rcParams['font.size'] = 14  # Set colorbar font size
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("top", size="3%", pad=0.3)
    plt.colorbar(im, ax=ax2, cax=cax2, orientation='horizontal')
    plt.subplots_adjust(bottom=0.08, top=0.98, left=0.11, right=0.99)

    fig.tight_layout()

    plt.show()



