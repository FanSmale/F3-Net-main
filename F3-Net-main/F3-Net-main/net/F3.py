"""
FWI-GISP

Created on Jan 2025

@author: Dora Liu
"""


from skimage.metrics import structural_similarity as ssim
from func.CannyLoss import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt


class CrossAttentionFusion(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()

        inter_channels = channels // reduction

        # ===== Q K V 投影 =====
        self.q_proj = nn.Conv2d(channels, inter_channels, 1)
        self.k_proj = nn.Conv2d(channels, inter_channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)

        # ===== 输出融合 =====
        self.out_proj = nn.Conv2d(channels, channels, 1)

        # ===== 可学习缩放（很关键🔥）=====
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, freq, spatial):
        """
        freq:    (B, C, H, W)  → Query
        spatial: (B, C, H, W)  → Key / Value
        """

        B, C, H, W = freq.shape

        # ===== 1️⃣ 投影 =====
        Q = self.q_proj(freq)        # (B, C', H, W)
        K = self.k_proj(spatial)     # (B, C', H, W)
        V = self.v_proj(spatial)     # (B, C,  H, W)

        # ===== 2️⃣ reshape =====
        Q = Q.view(B, -1, H * W).permute(0, 2, 1)   # (B, HW, C')
        K = K.view(B, -1, H * W)                    # (B, C', HW)
        V = V.view(B, -1, H * W)                    # (B, C,  HW)

        # ===== 3️⃣ attention map =====
        attn = torch.bmm(Q, K)                      # (B, HW, HW)
        attn = attn / (Q.shape[-1] ** 0.5)          # scaling
        attn = F.softmax(attn, dim=-1)

        # ===== 4️⃣ 加权 =====
        out = torch.bmm(V, attn.permute(0, 2, 1))   # (B, C, HW)
        out = out.view(B, C, H, W)

        # ===== 5️⃣ 残差融合（关键🔥）=====
        out = self.gamma * out + freq

        # ===== 6️⃣ 输出压缩 =====
        out = self.out_proj(out)

        return out

class ChannelAttentionFusion(nn.Module):
    def __init__(self, nf_in=256, nf_out=128):
        """
        通道注意力融合模块：将 fft_features 通道数 nf_in → nf_out
        """
        super(ChannelAttentionFusion, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(nf_in, nf_in // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf_in // 4, nf_in, 1, bias=False),
            nn.Sigmoid()
        )
        # 如果需要输出通道不同，还需要一个 1x1 卷积
        self.channel_reduce = nn.Conv2d(nf_in, nf_out, 1, bias=False)

    def forward(self, fft_features):
        # 生成注意力权重
        attention_weights = self.fc(self.global_avg_pool(fft_features))
        # 应用注意力
        fused = fft_features * attention_weights
        # 通道降维
        fused = self.channel_reduce(fused)
        return fused

class ConvNeXtBlock(nn.Module):
    expansion = 4  # 输出通道是out_dim的4倍

    def __init__(self, in_dim, out_dim, stride=1, downsample=None):
        super().__init__()
        self.downsample = downsample
        self.stride = stride
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.expanded_dim = out_dim * self.expansion

        # 深度卷积，输入输出通道 = in_dim，groups = in_dim
        self.dwconv = nn.Conv2d(in_dim, in_dim, kernel_size=7, padding=3, groups=in_dim, stride=stride)
        self.norm = nn.InstanceNorm2d(in_dim)

        # Pointwise conv 1: 将通道扩展到 4*out_dim
        self.pwconv1 = nn.Conv2d(in_dim, 4 * out_dim, kernel_size=1)
        self.act = nn.GELU()

        # Pointwise conv 2: 将通道压缩回 expanded_dim = 4*out_dim (这里保持一致，或者按设计改)
        self.pwconv2 = nn.Conv2d(4 * out_dim, self.expanded_dim, kernel_size=1)

        # gamma参数做残差缩放
        self.gamma = nn.Parameter(torch.ones((self.expanded_dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # 如果空间尺寸不匹配，插值到shortcut尺寸
        if x.shape[2:] != shortcut.shape[2:]:
            x = F.interpolate(x, size=shortcut.shape[2:], mode='bilinear', align_corners=False)

        return shortcut + self.gamma * x

class SeisNetR(nn.Module):
    def __init__(self, block, layers, in_channels=5):
        super(SeisNetR, self).__init__()
        self.in_channels = 64  # 初始通道数（backbone 内部状态）

        # 初始卷积：只在高度方向下采样（stride=(2,1)）
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=(2,1), padding=(3,3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 替代原先的 maxpool：使用两层 Conv2d 下采样 (stride=(2,1))，以高度为主下采样
        # 这两层都是可学习的 conv（kernel 3x1, padding=(1,0)）来保持宽度 70 不变
        self.down_conv_a = nn.Conv2d(64, 64, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=False)
        self.down_bn_a = nn.BatchNorm2d(64)
        self.down_conv_b = nn.Conv2d(64, 64, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=False)
        self.down_bn_b = nn.BatchNorm2d(64)
        # 残差层（注意 stride 设置）
        # layer1 用 stride=(2,2)：在这里同时下采样高度和宽度 -> 从 (125,70) -> (62,35)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=(2,2))
        # layer2 & layer3 保持 stride=2（即 (2,2)），产生 skip3 和最深输出 x
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        # 需要注意：block.expansion 会影响输出通道
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            # 这里使用 1x1 下采样 conv，stride 可以是 int 或 tuple (h_stride, w_stride)
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始 conv
        x = self.conv1(x)            # -> [B,64,500,70]  (1000 -> 500)
        x = self.bn1(x)
        x = self.relu(x)

        # 两次 conv 下采样（height only）替代原 maxpool，使得高度逐步变为 250 -> 125
        x = self.down_conv_a(x)      # -> [B,64,250,70]
        x = self.down_bn_a(x)
        x = F.relu(x)

        x = self.down_conv_b(x)      # -> [B,64,125,70]  这是新的 skip1
        x = self.down_bn_b(x)
        x = F.relu(x)

        skip1 = x                    # [B,64,125,70]

        # layer1: stride=(2,2) -> 从 (125,70) -> (~62,35)  (skip2)
        skip2 = self.layer1(skip1)   # [B,256,62,35]

        # layer2 -> skip3: stride=2 -> (62,35) -> (~31,18)
        skip3 = self.layer2(skip2)   # [B,512,31,18]

        # layer3 -> x (deepest): stride=2 -> (31,18) -> (~15,9)
        x = self.layer3(skip3)       # [B,1024,15,9]

        return x, skip1, skip2, skip3



class LightASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(LightASPP, self).__init__()
        self.convs = nn.ModuleList()

        # 1x1 卷积
        self.convs.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        # 空洞卷积
        for rate in atrous_rates:
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        # 最终投影
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        outputs = [conv(x) for conv in self.convs]
        return self.project(torch.cat(outputs, dim=1))


############################################
# Optimized DeepLabV3 for FWI with Skip Connections #
############################################

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        if c != self.in_channels:
            raise ValueError(f"Input channels {c} don't match SEBlock's expected {self.in_channels}")

        y = self.global_avg(x)  # [b,c,1,1]
        y = self.fc(y)  # [b,c,1,1]
        return x * y

class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        # max_out = max_out.view(-1, ch_2, 1, 1)
        # print(x.shape)#x.shape为[2,256,14,14]
        # print(max_out.shape)#max_out.shape为[2,256,1,1]
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


class SCSEModule2d(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SCSEModule2d, self).__init__()

        # 通道注意力模块（Channel SE）
        self.channel_excitation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 输出形状: [B, C, 1, 1]
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.Tanh(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 空间注意力模块（Spatial SE）
        self.spatial_excitation = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        cse_weight = self.channel_excitation(x)  # [B, C, 1, 1]
        sse_weight = self.spatial_excitation(x)  # [B, 1, H, W]

        return x * cse_weight + x * sse_weight

# 🔹 子像素卷积模块（带BN+ReLU）
class SubPixelConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super(SubPixelConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2),
                              kernel_size=3, padding=1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class FGLMSA(nn.Module):
    def __init__(self, channels):
        super(FGLMSA, self).__init__()

        # ===== 方向注意力 =====
        self.conv_h = nn.Conv2d(channels, channels, 1)
        self.conv_w = nn.Conv2d(channels, channels, 1)
        self.sigmoid = nn.Sigmoid()

        # ===== 可学习高频 =====
        self.high_freq = nn.Conv2d(
            channels, channels,
            kernel_size=3,
            padding=1,
            groups=channels
        )

        # ===== Sobel梯度 =====
        self.sobel_x = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.sobel_y = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)

        self._init_sobel()

        # ===== 融合权重 =====
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def _init_sobel(self):
        sobel_x = torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[-1,-2,-1],[0,0,0],[1,2,1]]], dtype=torch.float32)

        sobel_x = sobel_x.unsqueeze(0).repeat(self.sobel_x.weight.shape[0],1,1,1)
        sobel_y = sobel_y.unsqueeze(0).repeat(self.sobel_y.weight.shape[0],1,1,1)

        with torch.no_grad():
            self.sobel_x.weight.copy_(sobel_x)
            self.sobel_y.weight.copy_(sobel_y)

    def forward(self, x):

        # ===== 方向注意力 =====
        y_h = x.mean(dim=3, keepdim=True)
        y_w = x.mean(dim=2, keepdim=True)

        a_h = self.sigmoid(self.conv_h(y_h))
        a_w = self.sigmoid(self.conv_w(y_w))

        x = x * a_h * a_w + x

        # ===== 高频 =====
        high = self.high_freq(x)

        # ===== 梯度 =====
        grad_x = self.sobel_x(x)
        grad_y = self.sobel_y(x)
        grad = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

        # ===== 频域引导 =====
        fft = torch.fft.fft2(x)
        amp = torch.abs(fft)
        freq_weight = torch.sigmoid(amp.mean(dim=(2,3), keepdim=True))

        # ===== 融合 =====
        out = x + self.alpha * high + self.beta * grad
        out = out * freq_weight

        return out

class LMSA2D(nn.Module):
    """
    Local Multi-Scale Attention (EFMS-net style)

    功能：
    1. 方向注意力（H/W）
    2. Gaussian平滑 + 下采样 + 上采样
    3. 高频差分增强

    输入:  (B, C, H, W)
    输出:  (B, C, H, W)
    """

    def __init__(self, channels):
        super(LMSA2D, self).__init__()

        self.channels = channels

        # ===== 方向注意力 =====
        self.conv_h = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_w = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # ===== Gaussian 平滑（Depthwise）=====
        self.gaussian = nn.Conv2d(
            channels, channels,
            kernel_size=5,
            padding=2,
            groups=channels,
            bias=False
        )

        self._init_gaussian()

    # ---------------- 初始化 Gaussian ----------------
    def _init_gaussian(self):
        kernel_size = 5
        sigma = 1.0

        ax = torch.arange(kernel_size).float() - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')

        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()

        kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1,1,5,5)
        kernel = kernel.repeat(self.channels, 1, 1, 1)

        with torch.no_grad():
            self.gaussian.weight.copy_(kernel)
            self.gaussian.weight.requires_grad = False

    # ---------------- forward ----------------
    def forward(self, x):
        B, C, H, W = x.shape

        # ===== 1️⃣ 方向注意力 =====
        y_h = x.mean(dim=3, keepdim=True)  # (B,C,H,1)
        y_w = x.mean(dim=2, keepdim=True)  # (B,C,1,W)

        a_h = self.sigmoid(self.conv_h(y_h))
        a_w = self.sigmoid(self.conv_w(y_w))

        x_att = x * a_h * a_w + x   # residual增强

        # ===== 2️⃣ Gaussian平滑 =====
        smooth = self.gaussian(x_att)

        # ===== 3️⃣ 下采样（模拟低频）=====
        down = smooth[:, :, ::2, ::2]

        # ===== 4️⃣ 上采样回原尺寸 =====
        up = F.interpolate(
            down,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )

        # ===== 5️⃣ 高频增强 =====
        high_freq = x_att - up

        # ❗ 原论文更接近这种：直接输出高频
        return high_freq

class FFT_Process(nn.Module):
    def __init__(self, nf):
        super(FFT_Process, self).__init__()
        self.conv_channel_reduce = nn.Conv2d(1024, 64, kernel_size=1, stride=1, padding=0)
        # Preprocessing for frequency domain
        self.nf = nf
        self.freq_preprocess = nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0)
        self.feature_fusion = nn.Conv2d(nf * 2, nf, kernel_size=1, stride=1, padding=0)
        self.process_amp = self._make_process_block(nf)
        self.process_pha = self._make_process_block(nf)
        self.process_fr = self._make_process_block(nf)
        self.process_map = self._make_process_block(nf)

    def _make_process_block(self, nf):
        return nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0)
        )

    def _make_fr_map_block(self, x, nf):
        B, C, H, W = x.shape
        if C == nf:
            return x
        elif C < nf:
            repeat_factor = nf // C + (1 if nf % C != 0 else 0)
            x = x.repeat(1, repeat_factor, 1, 1)[:, :nf, :, :]
        else:
            x = torch.nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
            x = x[:, :nf, :, :]
        return x


    def multiply_and_softmax(self, vis, fra):
        # Normalize features to avoid numerical instability
        vis = F.normalize(vis, dim=1)
        fra = F.normalize(fra, dim=1)

        # Flatten and multiply
        features1_flattened = vis.view(vis.size(0), vis.size(1), -1)
        features2_flattened = fra.view(fra.size(0), fra.size(1), -1)
        multiplied = torch.mul(features1_flattened, features2_flattened)

        # Apply softmax
        multiplied_softmax = torch.softmax(multiplied, dim=2)
        multiplied_softmax = multiplied_softmax.view(vis.size(0), vis.size(1), vis.size(2), vis.size(3))

        # Residual connection
        vis_map = vis * multiplied_softmax + vis
        return vis_map

    def forward(self, x, fr, y_map):
        # ---------------- 输入检查 ----------------
        if fr is Ellipsis:
            raise ValueError("fr is not a valid tensor")
        # print(f"[FFT_Process] Input fr.shape: {fr.shape}")
        # print(f"[FFT_Process] Input x.shape: {x.shape}, y_map.shape: {y_map.shape}")

        # ---------------- 频域预处理 ----------------
        x_pre = self.freq_preprocess(x)
        # print(f"[FFT_Process] freq_preprocess x.shape: {x_pre.shape}")

        # ---------------- FFT ----------------
        x_freq = torch.fft.rfft2(x_pre, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        # print(f"[FFT_Process] After rfft2 -> mag.shape: {mag.shape}, pha.shape: {pha.shape}")

        # ---------------- 幅度 & 相位处理 ----------------
        mag = self.process_amp(mag)
        pha = self.process_pha(pha)
        # print(f"[FFT_Process] After process_amp/pha -> mag.shape: {mag.shape}, pha.shape: {pha.shape}")

        # ---------------- fr 处理 ----------------
        fr = self.process_fr(fr)
        # print(f"[FFT_Process] After process_fr -> fr.shape: {fr.shape}")

        # ---------------- 交叉注意力 ----------------
        # 自动对齐 fr 与 pha 尺寸
        if fr.shape[2:] != pha.shape[2:]:
            fr = F.interpolate(fr, size=pha.shape[2:], mode='bilinear', align_corners=False)
            # print(f"[FFT_Process] After align fr -> {fr.shape}")

        pha = self.multiply_and_softmax(pha, fr)
        # print(f"[FFT_Process] After multiply_and_softmax -> pha.shape: {pha.shape}")

        # ---------------- 注意力 map ----------------
        y_map = torch.sigmoid(self.process_map(y_map))
        # 自动对齐 y_map 与 mag 尺寸
        if y_map.shape[2:] != mag.shape[2:]:
            y_map = F.interpolate(y_map, size=mag.shape[2:], mode='bilinear', align_corners=False)
            # print(f"[FFT_Process] After align y_map -> {y_map.shape}")

        mag = mag * y_map + mag
        # print(f"[FFT_Process] After applying y_map -> mag.shape: {mag.shape}")

        # ---------------- 频域重建 ----------------
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        # IRFFT 回原始空间尺寸
        x_out = torch.fft.irfft2(x_out, s=(x.shape[2], x.shape[3]), norm='backward')
        # print(f"[FFT_Process] After irfft2 -> x_out.shape: {x_out.shape}")

        # ---------------- 残差连接 ----------------
        x_out_ff = x_out + x
        # print(f"[FFT_Process] Output x_out_ff.shape: {x_out_ff.shape}")

        return x_out_ff, fr, y_map

class FEP_Module(nn.Module):
    def __init__(self, in_channels):
        super(FEP_Module, self).__init__()

        # 能量分支
        self.energy_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1)
        )

        # 频率分支（幅度）
        self.freq_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1)
        )

        # 相位分支
        self.phase_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1)
        )

        # 融合
        self.fusion = nn.Sequential(
            nn.Conv2d(48, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, 1),  # 输出和输入通道一致
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # ===== 能量分支 =====
        energy = torch.mean(x**2, dim=1, keepdim=True)  # [B,1,H,W]
        energy_feat = self.energy_conv(energy)

        # ===== FFT 分支 =====
        xf = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(xf)
        phase = torch.angle(xf)

        # 插值到原始尺寸
        mag = F.interpolate(mag, size=(H, W), mode='bilinear', align_corners=False)
        phase = F.interpolate(phase, size=(H, W), mode='bilinear', align_corners=False)

        freq_feat = self.freq_conv(mag)
        phase_feat = self.phase_conv(phase)

        # ===== 融合 =====
        fusion = torch.cat([energy_feat, freq_feat, phase_feat], dim=1)  # [B,48,H,W]
        attn = self.fusion(fusion)  # [B,C,H,W] 和输入通道一致
        return attn


class ASPPGate(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # Channel attention
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

        # Spatial attention
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        ch = self.channel_gate(x)
        sp = self.spatial_gate(x)
        return x * ch * sp



# ---------------- 轻量卷积下采样 ----------------
class DownsampleConv(nn.Module):
    """轻量下采样卷积，尺寸减半，同时保持通道"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class F3Net(nn.Module):
    def __init__(self, in_channels=5, n_classes=1, nf=64, atrous_rates=[6, 12]):
        super(F3Net, self).__init__()

        # ---------------- Backbone ----------------
        self.backbone = SeisNetR(ConvNeXtBlock, [3, 4, 6], in_channels=in_channels)

        # ---------------- FFT / FEP for skips ----------------
        self.fft_skip1 = FFT_Process(nf=64)
        self.fft_skip2 = FFT_Process(nf=64)
        self.fft_skip3 = FFT_Process(nf=128)

        self.fep_skip1 = FEP_Module(in_channels=64)
        self.fep_skip2 = FEP_Module(in_channels=64)
        self.fep_skip3 = FEP_Module(in_channels=128)

        self.conv_fr_skip1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_fr_skip2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_fr_skip3 = nn.Conv2d(128, 128, 3, 1, 1)

        # ---------------- 注意力模块 ----------------
        self.attention_high = CBAMLayer(1024)

        # ---------------- ASPP ----------------
        self.aspp = LightASPP(1024, 128, atrous_rates)
        self.aspp_gate = ASPPGate(128)

        # ---------------- Channel Attention Fusion ----------------
        self.channel_attention_fusion = ChannelAttentionFusion(nf_in=64+64+128, nf_out=128)

        # ---------------- LMSA / FGLMSA ----------------
        # concat(skip_fused + ASPP) = 128 + 128 = 256
        self.lmsa = LMSA2D(256)
        self.fglmsa = FGLMSA(256)
        self.use_fglmsa = True
        self.use_lmsa = False

        # ---------------- Decoder ----------------
        self.decoder_stage1 = nn.Sequential(
            SubPixelConvBlock(256, 128, upscale_factor=3),
            nn.Conv2d(128 + 512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.decoder_stage2 = nn.Sequential(
            SubPixelConvBlock(256, 128, upscale_factor=3),
            nn.Conv2d(128 + 256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.decoder_stage3 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, 1)
        )

        # ---------------- skip 降维 ----------------
        self.channel_reduce_skip2 = nn.Conv2d(256, 64, 1)
        self.channel_reduce_skip3 = nn.Conv2d(512, 128, 1)

    # ---------------- extract Fourier feature ----------------
    def extract_fr(self, x, name):
        fr = torch.fft.rfft2(x, norm='backward')
        fr = torch.angle(fr)
        if name == "skip1":
            fr = self.conv_fr_skip1(fr)
        elif name == "skip2":
            fr = self.conv_fr_skip2(fr)
        elif name == "skip3":
            fr = self.conv_fr_skip3(fr)
        elif name == "aspp":
            fr = self.conv_fr_aspp(fr)
        else:
            raise ValueError(name)
        return fr

    # ---------------- process skip ----------------
    def process_skip(self, skip, name, target_size):
        if name == "skip2":
            skip = self.channel_reduce_skip2(skip)
        elif name == "skip3":
            skip = self.channel_reduce_skip3(skip)

        fep = getattr(self, f"fep_{name}")
        fft_module = getattr(self, f"fft_{name}")

        fr = self.extract_fr(skip, name)
        y_map = fep(skip)
        x_out, fr_out, y_map_out = fft_module(skip, fr, y_map)

        if x_out.shape[2:] != target_size:
            x_out = F.interpolate(x_out, size=target_size, mode='bilinear', align_corners=False)
        return x_out

    # ---------------- Forward ----------------
    def forward(self, input_x, label_dsp_dim=(70, 70), return_features=False):
        # Backbone
        x, skip1, skip2, skip3 = self.backbone(input_x)

        # ASPP + Attention
        x_aspp = self.aspp(self.attention_high(x))
        fusion_size = x_aspp.shape[2:]

        # ---------------- FFT + FEP for skips ----------------
        fft_skip1 = self.process_skip(skip1, "skip1", fusion_size)
        fft_skip2 = self.process_skip(skip2, "skip2", fusion_size)
        fft_skip3 = self.process_skip(skip3, "skip3", fusion_size)

        # ---------------- Channel Attention Fusion for skips ----------------
        skip_fused = self.channel_attention_fusion(
            fft_features=torch.cat([fft_skip1, fft_skip2, fft_skip3], dim=1)
        )

        # ⭐ 不做 FFT
        aspp_feat = self.aspp_gate(x_aspp)

        # ---------------- concat ----------------
        multi_scale = torch.cat([skip_fused, aspp_feat], dim=1)

        # ---------------- LMSA / FGLMSA ----------------
        if self.use_fglmsa:
            multi_scale = self.fglmsa(multi_scale)
        elif self.use_lmsa:
            multi_scale = self.lmsa(multi_scale)
        else:
            #multi_scale = self.lmsa(multi_scale)
            multi_scale = multi_scale

        # ---------------- Decoder ----------------
        x = self.decoder_stage1[0](multi_scale)
        x = F.interpolate(x, size=skip3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip3], dim=1)
        x = self.decoder_stage1[1:](x)

        x = self.decoder_stage2[0](x)
        x = F.interpolate(x, size=skip2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip2], dim=1)
        x = self.decoder_stage2[1:](x)

        x = F.interpolate(x, size=skip1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip1], dim=1)
        last = x
        x = self.decoder_stage3(x)

        # 输出对齐
        if x.shape[2:] != label_dsp_dim:
            x = F.interpolate(x, size=label_dsp_dim, mode='bilinear', align_corners=False)

        if return_features:
            return {"last": last}
        else:
            return x

############################
# SSIM 模块
############################
def gaussian_window(window_size, sigma, channel):
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    g = g[:, None] * g[None, :]
    window = g.expand(channel, 1, window_size, window_size).contiguous()
    return window

class SSIM(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, channel=1, eps=1e-6):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.eps = eps
        self.channel = channel
        self.register_buffer("window", gaussian_window(window_size, sigma, channel))

    def forward(self, x, y):
        window = self.window.to(x.device).type_as(x)

        mu_x = F.conv2d(x, window, padding=self.window_size // 2, groups=self.channel)
        mu_y = F.conv2d(y, window, padding=self.window_size // 2, groups=self.channel)

        sigma_x = F.conv2d(x * x, window, padding=self.window_size // 2, groups=self.channel) - mu_x ** 2
        sigma_y = F.conv2d(y * y, window, padding=self.window_size // 2, groups=self.channel) - mu_y ** 2
        sigma_xy = F.conv2d(x * y, window, padding=self.window_size // 2, groups=self.channel) - mu_x * mu_y

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2) + self.eps)

        return ssim_map.mean()


############################
# 内部一致性损失 (L2 + SSIM)
############################
class SPLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, channel=1):
        """
        alpha: L2 权重
        beta: SSIM 权重
        channel: 输入通道数 (通常 1)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ssim = SSIM(channel=channel)

    def forward(self, pred, target):
        l2_loss = F.mse_loss(pred, target)
        ssim_loss = 1 - self.ssim(pred, target)  # SSIM 越大越好，所以取 1-SSIM
        return self.alpha * l2_loss + self.beta * ssim_loss

def visualize_loss_maps(pred, target, edge_loss_module, alpha=1.0, beta=1.0, sample_idx=0):
    """
    可视化：边缘图 + 内部加权图
    pred: 网络预测 [B,1,H,W]
    target: 真实速度 [B,1,H,W]
    edge_loss_module: MultiScaleEdgeWeightedL2Loss 实例
    alpha, beta: SPLoss 权重
    sample_idx: 可视化哪一张样本
    """
    # 取单样本
    pred_s = pred[sample_idx:sample_idx+1]
    target_s = target[sample_idx:sample_idx+1]

    # ===== 边缘图 =====
    with torch.no_grad():
        # fine, coarse, combined
        fine = edge_loss_module._gradient_mag(target_s)
        coarse = edge_loss_module._gradient_mag(F.avg_pool2d(target_s, 3, 1, 1))
        fine = fine / fine.amax(dim=(2,3), keepdim=True).clamp_min(1e-6)
        coarse = coarse / coarse.amax(dim=(2,3), keepdim=True).clamp_min(1e-6)
        combined_edge = (1 - edge_loss_module.coarse_weight) * fine + edge_loss_module.coarse_weight * coarse
        weighted_edge_map = 1.0 + edge_loss_module.lam * combined_edge

    # ===== 内部加权图 (SPLoss: L2 + 1-SSIM) =====
    l2_map = (pred_s - target_s)**2
    ssim_module = SPLoss(channel=1).ssim  # 调用SSIM模块
    try:
        ssim_map = ssim_module.compute_map(pred_s, target_s)
    except:
        ssim_map = torch.ones_like(pred_s) * ssim_module(pred_s, target_s)
    ssim_err_map = 1 - ssim_map
    weighted_inside_map = alpha * l2_map + beta * ssim_err_map

    # ===== 可视化 =====
    plt.figure(figsize=(16,4))

    plt.subplot(1,5,1)
    plt.imshow(fine[0,0].cpu().numpy(), cmap='viridis')
    plt.title("Fine Edge Map")
    plt.colorbar()

    plt.subplot(1,5,2)
    plt.imshow(coarse[0,0].cpu().numpy(), cmap='viridis')
    plt.title("Coarse Edge Map")
    plt.colorbar()

    plt.subplot(1,5,3)
    plt.imshow(combined_edge[0,0].cpu().numpy(), cmap='viridis')
    plt.title("Combined Edge Map")
    plt.colorbar()

    plt.subplot(1,5,4)
    plt.imshow(weighted_edge_map[0,0].cpu().numpy(), cmap='magma')
    plt.title("Weighted Edge Loss Map")
    plt.colorbar()

    plt.subplot(1,5,5)
    plt.imshow(weighted_inside_map[0,0].cpu().numpy(), cmap='magma')
    plt.title("Weighted Internal Loss Map")
    plt.colorbar()

    plt.show()

############################
# 多尺度边界加权 L2 损失 (无平滑)
############################
class MultiScaleEdgeWeightedL2Loss(nn.Module):
    def __init__(self, lam=1.0, eps=1e-6, coarse_weight=0.3):
        """
        lam: 边缘权重强度 (0.5~3.0 推荐)
        coarse_weight: 大尺度边缘占比 (0~1)
        """
        super().__init__()
        self.lam = lam
        self.eps = eps
        self.coarse_weight = coarse_weight

        sx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        sy = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32)
        self.register_buffer("sobel_x", sx.view(1,1,3,3))
        self.register_buffer("sobel_y", sy.view(1,1,3,3))
    def _gradient_mag(self, x):
        C = x.shape[1]
        kx = self.sobel_x.repeat(C, 1, 1, 1).to(x.device).type(x.dtype)
        ky = self.sobel_y.repeat(C, 1, 1, 1).to(x.device).type(x.dtype)

        gx = F.conv2d(x, kx, padding=1, groups=C)
        gy = F.conv2d(x, ky, padding=1, groups=C)
        return torch.sqrt(gx * gx + gy * gy + self.eps)

    @torch.no_grad()
    def _edge_map(self, x, visualize=True):
        fine = self._gradient_mag(x)
        coarse = self._gradient_mag(F.avg_pool2d(x, kernel_size=3, stride=1, padding=1))

        fine = fine / fine.amax(dim=(2, 3), keepdim=True).clamp_min(self.eps)
        coarse = coarse / coarse.amax(dim=(2, 3), keepdim=True).clamp_min(self.eps)

        edge = (1 - self.coarse_weight) * fine + self.coarse_weight * coarse
        if visualize:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(fine[0, 0].cpu().numpy(), cmap='viridis')
            plt.title("Fine Edge Map")
            plt.colorbar()
            plt.subplot(1, 3, 2)
            plt.imshow(coarse[0, 0].cpu().numpy(), cmap='viridis')
            plt.title("Coarse Edge Map")
            plt.colorbar()
            plt.subplot(1, 3, 3)
            plt.imshow(edge[0, 0].cpu().numpy(), cmap='viridis')
            plt.title("Combined Edge Map")
            plt.colorbar()
            plt.show()
        return edge

    def forward(self, pred, target, return_map=False, visualize=False):
        err = torch.abs(pred - target)  # L1 误差
        edge = self._edge_map(target, visualize=visualize)
        w = 1.0 + self.lam * edge
        loss = (w * err).sum() / (w.sum() + self.eps)

        if return_map:
            return loss, w  # 同时返回权重图
        else:
            return loss


############################
# 动态混合损失
############################
class DynamicHybridLoss(nn.Module):
    def __init__(self, lam=1.0, alpha=1.0, beta=1.0,
                 start_balance=0.2, end_balance=0.8, max_epochs=100, channel=1):
        """
        动态混合损失函数：训练初期重内部(L2+SSIM)，后期重边界(EdgeWeighted)
        """
        super().__init__()
        self.edge_loss = MultiScaleEdgeWeightedL2Loss(lam=lam)
        self.sp_loss = SPLoss(alpha=alpha, beta=beta, channel=channel)
        self.start_balance = start_balance
        self.end_balance = end_balance
        self.max_epochs = max_epochs
        self.cur_epoch = 0

    def update_epoch(self, epoch):
        self.cur_epoch = epoch

    def forward(self, pred, target):
        progress = min(self.cur_epoch / self.max_epochs, 1.0)
        balance = self.start_balance + (self.end_balance - self.start_balance) * \
                  (0.5 - 0.5 * math.cos(progress * math.pi))

        # loss_edge = self.edge_loss(pred, target)   # 边界
        loss_edge, w_map = self.edge_loss(pred, target, return_map=True)
        loss_inside = self.sp_loss(pred, target)   # 内部 (L2+SSIM)

        loss = balance * loss_edge + (1 - balance) * loss_inside
        return loss, balance, w_map

############################################
#                Main Function             #
############################################
if __name__ == '__main__':
    # 生成一个测试输入 (batch_size=2, 5通道, 400x310)
    x = torch.zeros((5, 5, 1000, 70))
    model = F3Net(in_channels=5, n_classes=1)
    label_dsp_dim = [70, 70]
    # 计算输出
    output = model(x, label_dsp_dim)
    print(f"Output shape: {output.size()}")  # 预期: (2, 1, 201, 301)