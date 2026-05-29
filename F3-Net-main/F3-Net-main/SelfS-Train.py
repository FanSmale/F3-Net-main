
"""
Created on 2023/10/17 10:13
@author: XUQIONG
"""

import os
import time
from net.F3 import *

from param_config import *
from data.data import *
from func.utils import *
from data.utils import *
from path_config import *
from func.datasets_reader import *
from net.F3 import DynamicHybridLoss

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -----------------------------
# GPU设置
# -----------------------------
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

# -----------------------------
# 网络和优化器
# -----------------------------
net = F3Net()  # 可替换 InversionNet | ABA_Net|Model

#net = SEGDeepLab()
net = net.to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# -----------------------------
# 预训练模型加载
# -----------------------------
if ReUse:
    print('***************** Loading pre-training model *****************')
    premodel_file = os.path.join(results_dir, PreModelname)
    net.load_state_dict(torch.load(premodel_file))
    net = net.to(device)
    print('Finish loading:', premodel_file)
# -----------------------------
# 数据加载
# -----------------------------
print('***************** Loading training dataset *****************')
dataset_dir = data_dir

use_mat_data = False

if use_mat_data:
    # 训练集
    trainSet = DatasetSEG(dataset_dir,
                           train_size,
                           1,
                           "train")

    valSet = DatasetSEG(dataset_dir,
                              ValSize,
                              1,#1601
                              "test")
else:
    trainSet = Dataset_openfwi(dataset_dir, train_size, 1, "seismic", "train")
    valSet   = Dataset_openfwi(dataset_dir, ValSize, 1, "seismic", "test")

#train_loader = torch.utils.data.DataLoader(trainSet, batch_size=train_batch_size, shuffle=True)
train_loader = DataLoader(trainSet, batch_size=train_batch_size, shuffle=True)
#val_loader = torch.utils.data.DataLoader(valSet, batch_size=train_batch_size)
val_loader = DataLoader(valSet, batch_size=train_batch_size, shuffle=True)



# -----------------------------
# 损失函数初始化
# -----------------------------
criterion = DynamicHybridLoss(
    lam=0.5,       # 边界权重
    alpha=1,     # L2 权重
    beta=0.5,      # SSIM 权重
    start_balance=0.2,
    end_balance=0.8,
    max_epochs=thirdstage_epochs,
    channel=1
).to(device)

# -----------------------------
# 损失列表初始化
# -----------------------------
train_loss_list = np.array([])
val_loss_list = np.array([])
balance_list = np.array([])   # 🔥 记录边界 vs 内部的权重平衡

SAVE_INTERVAL = 1  # 每隔多少个 epoch 保存一次热力图
SAVE_DIR = "./vis_dsp_lossmap"
os.makedirs(SAVE_DIR, exist_ok=True)
# -----------------------------
# 训练函数
# -----------------------------
def train(epoch):
    total_loss = 0.0
    net.train()


    for batch_idx, (seismic_datas, vmodels) in enumerate(train_loader):
        batch_start = time.time()

        # # 网络输出检查（forward 后立即打印）
        # with torch.no_grad():
        #     outputs = net(seismic_datas.to(device), vmodels.shape[2:])  # or label_dsp_dim
        #
        #
        # # 检查损失规模
        # loss, _ = criterion(outputs, vmodels.to(outputs.device))


        seismic_datas = seismic_datas[0].to(device)
        vmodels = vmodels[0].to(device)
        # seismic_datas = seismic_datas.to(device)
        # vmodels = vmodels.to(device)


        optimizer.zero_grad()

        if NoiseFlag:
            # 添加高斯噪声
            noise_mean = 0
            noise_std = 0.15 # 0.01 0.03 0.05 0.1
            noise = torch.normal(mean=noise_mean, std=noise_std, size=seismic_datas.shape).to(device)
            seismic_datas_noisy = seismic_datas + noise  # 先保留原变量
            # 假设数据是 seismic_datas_noisy[0, 2, :, :]
            # 可视化固定样本


            # # ===== 计算并打印噪声分贝（SNR）=====
            # signal_power = torch.mean(seismic_datas ** 2)
            # noise_power = noise_std ** 2  # 高斯噪声功率 = 方差
            # snr_db = 10 * torch.log10(signal_power / noise_power)
            # print(f"当前噪声标准差 {noise_std} 对应的信噪比约为: {snr_db.item():.2f} dB")

            # 替换原数据
            seismic_datas = seismic_datas_noisy

        label_dsp_dim = vmodels.shape[2:]

        # # ======== 生成 cond 张量 ========
        # # 假设每个样本有 5 个震源，x 坐标均匀分布在地震数据宽度上，z 坐标固定在地表 0
        # B = seismic_datas.shape[0]  # batch size
        # W = seismic_datas.shape[-1]  # seismic 数据宽度（采样点数）
        # num_shots = 5  # 震源数量

        # x_coords = torch.linspace(0, W - 1, num_shots, dtype=torch.float32, device=device)  # 均匀分布
        # z_coords = torch.zeros(num_shots, dtype=torch.float32, device=device)  # 地表 z=0
        #
        # single_cond = torch.stack([x_coords, z_coords], dim=1)  # [5,2]
        # cond_tensor = single_cond.unsqueeze(0).repeat(B, 1, 1)  # [B,5,2]
        # ================================

        outputs = net(seismic_datas, label_dsp_dim)
        #outputs = net(seismic_datas, cond_tensor, label_dsp_dim)
        #outputs = outputs.to(torch.float32)

        # -----------------------------
        # 损失函数计算 (动态更新)
        # -----------------------------
        criterion.update_epoch(epoch)
        loss, balance, w_map = criterion(outputs, vmodels)

        if np.isnan(loss.item()):
            raise ValueError("loss is nan")

        loss.backward()
        optimizer.step()



        total_loss += loss.item()
        batch_time = time.time() - batch_start

        # # 输出预测统计量
        # print(f"Debug - pred min/max/mean/std: "
        #       f"{outputs.min().item():.2f}/"
        #       f"{outputs.max().item():.2f}/"
        #       f"{outputs.mean().item():.2f}/"
        #       f"{outputs.std().item():.2f}")
        #
        # # 输出平均梯度
        # total_grad = 0.0
        # for name, param in net.named_parameters():
        #     if param.grad is not None:
        #         total_grad += param.grad.abs().mean().item()
        # print(f"Debug - avg grad: {total_grad:.6f}")

        if batch_idx % 10 == 0:
            print(f"[Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} | Balance(edge={balance:.3f}, inside={1-balance:.3f}) "
                  f"| BatchTime: {batch_time:.3f}s")

    avg_loss = total_loss / len(train_loader)
    return avg_loss, balance

# -----------------------------
# 验证函数
# -----------------------------
def validate(epoch):
    total_loss = 0.0
    net.eval()

    with torch.no_grad():
        for seismic_datas, vmodels in val_loader:
            seismic_datas = seismic_datas[0].to(device)
            vmodels = vmodels[0].to(device)
            # seismic_datas = seismic_datas.to(device)
            # vmodels = vmodels.to(device)


            label_dsp_dim = vmodels.shape[2:]

            # # ======== 生成 cond 张量 ========
            # # 假设每个样本有 5 个震源，x 坐标均匀分布在地震数据宽度上，z 坐标固定在地表 0
            # B = seismic_datas.shape[0]  # batch size
            # W = seismic_datas.shape[-1]  # seismic 数据宽度（采样点数）
            # num_shots = 5  # 震源数量
            #
            # x_coords = torch.linspace(0, W - 1, num_shots, dtype=torch.float32, device=device)  # 均匀分布
            # z_coords = torch.zeros(num_shots, dtype=torch.float32, device=device)  # 地表 z=0
            #
            # single_cond = torch.stack([x_coords, z_coords], dim=1)  # [5,2]
            # cond_tensor = single_cond.unsqueeze(0).repeat(B, 1, 1)  # [B,5,2]
            # # ================================


            outputs = net(seismic_datas, label_dsp_dim)
            #outputs = net(seismic_datas, cond_tensor, label_dsp_dim)
            #outputs = outputs.to(torch.float32)

            criterion.update_epoch(epoch)
            loss, _ , _ = criterion(outputs, vmodels)

            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss

# -----------------------------
# 训练主循环
# -----------------------------
print(f"Training size: {train_size}, Batch size: {train_batch_size}, Epochs: {thirdstage_epochs}, LR: {learning_rate:.5f}")
start = time.time()

for epoch in range(thirdstage_epochs):
    epoch_start = time.time()

    train_loss, balance = train(epoch)
    val_loss = validate(epoch)

    train_loss_list = np.append(train_loss_list, train_loss)
    val_loss_list = np.append(val_loss_list, val_loss)
    balance_list = np.append(balance_list, balance)

    print(f"Epoch [{epoch+1}/{thirdstage_epochs}], "
          f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, "
          f"Balance(edge={balance:.3f}, inside={1-balance:.3f})")
    print(f"Epoch time: {(time.time() - epoch_start)//60:.0f}m {(time.time() - epoch_start)%60:.0f}s")

    # -----------------------------
    # 每5个epoch可视化前5个SEG样本
    # -----------------------------
    # if (epoch + 1) % 1 == 0:
    #     net.eval()  # 切换到评估模式
    #     with torch.no_grad():
    #         samples_visualized = 0
    #         for seis_batch, gt_vmod_batch in train_loader:  # 遍历训练集
    #             if torch.cuda.is_available():
    #                 seis_batch = seis_batch.cuda(non_blocking=True)
    #                 gt_vmod_batch = gt_vmod_batch.cuda(non_blocking=True)
    #
    #             # 预测
    #             pred_batch = net(seis_batch, model_dim)  # (B,1,H,W)
    #
    #             batch_size = seis_batch.shape[0]
    #             for k in range(batch_size):
    #                 if samples_visualized >= 5:  # 只可视化前5个样本
    #                     break
    #
    #                 seis_sample = seis_batch[k, 0].cpu().numpy()
    #                 gt_vmod_sample = gt_vmod_batch[k, 0].cpu().numpy()
    #                 pred_vmod_sample = pred_batch[k, 0].cpu().numpy()
    #
    #                 min_velocity = torch.min(gt_vmod_batch[k]).item()
    #                 max_velocity = torch.max(gt_vmod_batch[k]).item()
    #
    #                 # 可视化
    #                 pain_seg_seismic_data(seis_sample)
    #                 pain_seg_velocity_model(gt_vmod_sample, min_velocity, max_velocity)
    #                 pain_seg_velocity_model(pred_vmod_sample, min_velocity, max_velocity)
    #
    #                 samples_visualized += 1
    #
    #             if samples_visualized >= 5:
    #                 break
    #
    #     net.train()  # 切回训练模式
    # -----------------------------
    # 每个 epoch 可视化前5个速度模型样本 + 三类边缘图
    # -----------------------------
    # SAVE_DIR = "./loss_maps_png"
    # os.makedirs(SAVE_DIR, exist_ok=True)
    #
    # import matplotlib.colors as mcolors
    # # 生成透明 colormap
    # def transparent_cmap(base_cmap='inferno', min_alpha=0.0, max_alpha=1.0):
    #     cmap = plt.get_cmap(base_cmap)
    #     cmap_colors = cmap(np.arange(cmap.N))
    #     alphas = np.linspace(min_alpha, max_alpha, cmap.N)
    #     cmap_colors[:, -1] = alphas
    #     return mcolors.ListedColormap(cmap_colors)
    #
    #
    # # 定义透明版 colormap
    # cmap_edge_trans = transparent_cmap('inferno', min_alpha=0.0, max_alpha=1.0)
    # cmap_loss_trans = transparent_cmap('magma', min_alpha=0.0, max_alpha=1.0)

    # # ================== 主体可视化逻辑 ==================
    # if (epoch + 1) % 1 == 0:
    #     net.eval()
    #     with torch.no_grad():
    #         samples_visualized = 0
    #         for seismic_datas, gt_vmod_batch in train_loader:
    #             seismic_datas = seismic_datas[0].to(device)
    #             gt_vmod_batch = gt_vmod_batch[0].to(device)
    #
    #             seis_input = seismic_datas[:, :5, :, :]
    #             pred_batch = net(seis_input, model_dim)
    #
    #             batch_size = seis_input.shape[0]
    #             for k in range(batch_size):
    #                 if samples_visualized >= 5:
    #                     break
    #
    #                 gt_vmodel_of_k = gt_vmod_batch[k, 0, :, :].cpu().numpy()
    #                 pd_vmodel_of_k = pred_batch[k, 0, :, :].cpu().numpy()
    #                 min_v, max_v = np.min(gt_vmodel_of_k), np.max(gt_vmodel_of_k)
    #
    #                 # --- 边缘图 ---
    #                 fine_map = criterion.edge_loss._gradient_mag(gt_vmod_batch[k:k + 1])
    #                 coarse_map = criterion.edge_loss._gradient_mag(
    #                     F.avg_pool2d(gt_vmod_batch[k:k + 1], 3, 1, 1)
    #                 )
    #                 fine_norm = fine_map / fine_map.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    #                 coarse_norm = coarse_map / coarse_map.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    #                 edge_map = (1 - criterion.edge_loss.coarse_weight) * fine_norm + \
    #                            criterion.edge_loss.coarse_weight * coarse_norm
    #                 weighted_edge_map = 1.0 + criterion.edge_loss.lam * edge_map
    #
    #                 # --- 内部加权图 ---
    #                 l2_map = (pred_batch[k:k + 1] - gt_vmod_batch[k:k + 1]) ** 2
    #                 ssim_module = SPLoss(channel=1).ssim
    #                 try:
    #                     ssim_map = ssim_module.compute_map(pred_batch[k:k + 1], gt_vmod_batch[k:k + 1])
    #                 except:
    #                     ssim_map = torch.ones_like(pred_batch[k:k + 1]) * ssim_module(
    #                         pred_batch[k:k + 1], gt_vmod_batch[k:k + 1])
    #                 ssim_err_map = 1 - ssim_map
    #                 weighted_inside_map = 1.0 * l2_map + 1.0 * ssim_err_map
    #
    #                 # --- 可视化并保存 PNG ---
    #                 fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    #                 fig.patch.set_alpha(0.0)  # 整体透明背景
    #
    #                 # 第一行：速度模型
    #                 axes[0, 0].imshow(gt_vmodel_of_k, cmap='viridis', vmin=min_v, vmax=max_v)
    #                 axes[0, 0].set_title("GT Velocity");
    #                 axes[0, 0].axis("off")
    #
    #                 axes[0, 1].imshow(pd_vmodel_of_k, cmap='viridis', vmin=min_v, vmax=max_v)
    #                 axes[0, 1].set_title("Pred Velocity");
    #                 axes[0, 1].axis("off")
    #
    #                 axes[0, 2].axis("off")
    #
    #                 # 第二行：透明边缘图
    #                 axes[1, 0].imshow(fine_norm[0, 0].cpu().numpy(), cmap=cmap_edge_trans)
    #                 axes[1, 0].set_title("Fine Edge Map");
    #                 axes[1, 0].axis("off")
    #
    #                 axes[1, 1].imshow(coarse_norm[0, 0].cpu().numpy(), cmap=cmap_edge_trans)
    #                 axes[1, 1].set_title("Coarse Edge Map");
    #                 axes[1, 1].axis("off")
    #
    #                 axes[1, 2].imshow(edge_map[0, 0].cpu().numpy(), cmap=cmap_edge_trans)
    #                 axes[1, 2].set_title("Combined Edge Map");
    #                 axes[1, 2].axis("off")
    #
    #                 # 第三行：透明内部损失图
    #                 axes[2, 0].imshow(l2_map[0, 0].cpu().numpy(), cmap=cmap_loss_trans)
    #                 axes[2, 0].set_title("L2 Map");
    #                 axes[2, 0].axis("off")
    #
    #                 axes[2, 1].imshow(ssim_err_map[0, 0].cpu().numpy(), cmap=cmap_loss_trans, vmin=0, vmax=0.1)
    #                 axes[2, 1].set_title("1-SSIM Map");
    #                 axes[2, 1].axis("off")
    #
    #                 axes[2, 2].imshow(weighted_inside_map[0, 0].cpu().numpy(), cmap=cmap_loss_trans)
    #                 axes[2, 2].set_title("Weighted Internal Loss");
    #                 axes[2, 2].axis("off")
    #
    #                 plt.tight_layout()
    #
    #                 # --- 保存 PNG（透明背景） ---
    #                 save_path = os.path.join(SAVE_DIR, f"epoch{epoch + 1}_sample{k + 1}_trans.png")
    #                 plt.savefig(save_path, dpi=150, transparent=True)
    #                 plt.close(fig)
    #
    #                 samples_visualized += 1
    #
    #             if samples_visualized >= 5:
    #                 break
    #     net.train()

    # 保存模型
    if (epoch + 1) % SaveEpoch == 0:
        save_path = os.path.join(results_dir, f"{ModelName}_{dataset_name}_epoch{epoch+1}.pkl")
        torch.save(net.state_dict(), save_path)
        print(f"Model saved: {save_path}")

# -----------------------------
# 训练总耗时
# -----------------------------
total_time = time.time() - start
print(f"Training complete in {total_time//60:.0f}m {total_time%60:.0f}s")

# -----------------------------
# 保存训练和验证损失
# -----------------------------
font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 17}
font3 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 21}

SaveTrainValidResults(train_loss=train_loss_list,
                      val_loss=val_loss_list,
                      SavePath=models_dir,
                      ModelName=ModelName,
                      font2=font2,
                      font3=font3)

#🔥 如果你想额外可视化 balance 曲线，可以在这里加:
# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(balance_list, label="Edge Balance")
# plt.plot(1-balance_list, label="Inside Balance")
# plt.xlabel("Epoch", fontdict=font2)
# plt.ylabel("Weight", fontdict=font2)
# plt.legend(prop={'family': 'Times New Roman', 'size': 12})
# plt.title("Dynamic Loss Balance", fontdict=font3)
#
# # 设置坐标轴刻度字体
# plt.xticks(fontname='Times New Roman', fontsize=12)
# plt.yticks(fontname='Times New Roman', fontsize=12)
#
# plt.savefig(os.path.join(models_dir, f"{ModelName}_balance.png"), dpi=300, bbox_inches='tight')
# plt.close()
