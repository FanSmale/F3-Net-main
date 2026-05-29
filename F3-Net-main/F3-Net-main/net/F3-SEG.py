from func.CannyLoss import *
import math

from net.RSF import FEP_Module, FFT_Process, FGLMSA
from net.luminance_map import ChannelAttentionFusion


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


class BasicResidualBlock(nn.Module):
    expansion = 4  # 输出通道是 out_channels*expansion

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)
        return out

class SeisNetR(nn.Module):
    def __init__(self, block, layers, in_channels=29):
        super(SeisNetR, self).__init__()
        self.in_channels = 64
        self.relu = nn.ReLU(inplace=True)

        # ✅ DownConvA: [B,29,400,301] -> [B,64,200,150]
        self.down_conv_a = nn.Conv2d(in_channels, 64, kernel_size=7, stride=(2, 2), padding=3, bias=False)
        self.down_bn_a = nn.BatchNorm2d(64)

        # ✅ DownConvB: [B,64,200,150] -> [B,64,100,75]
        self.down_conv_b = nn.Conv2d(64, 64, kernel_size=3, stride=(2, 2), padding=1, bias=False)
        self.down_bn_b = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=(2, 2))

        # ✅ Layer2: deeper features, no downsample
        self.layer2 = self._make_layer(block, 128, layers[1], stride=(1, 1))

        # ✅ Layer3: deepest features, no downsample
        self.layer3 = self._make_layer(block, 256, layers[2], stride=(1, 1))

        # 🔹 下采样最深特征到 [25,19]
        self.downsample_x = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1, bias=False),  # 50x35 -> 25x18
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=(3,2), stride=1, padding=0, bias=False),  # 25x18 -> 25x19
            nn.ReLU(inplace=True)
        )

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # [B,29,400,301]
        x = self.down_conv_a(x)      # -> [B,64,200,150]
        x = self.down_bn_a(x)
        x = self.relu(x)
        skip1 = x

        x = self.down_conv_b(x)      # -> [B,64,100,75]
        x = self.down_bn_b(x)
        x = self.relu(x)

        skip2 = x

        x = self.layer1(x)           # -> [B,256,50,35]
        skip3 = x

        x = self.layer2(skip3)       # -> [B,512,50,35]
        x = self.layer3(x)           # -> [B,1024,50,35]

        # 🔹 用卷积下采样得到最深特征 [B,1024,25,19]
        x = self.downsample_x(x)     # -> [B,1024,25,19]

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

#FFT
class F3SEG(nn.Module):
    def __init__(self, in_channels=29, n_classes=1, atrous_rates=[6,12]):
        super().__init__()
        # Backbone: 注意我们现在传入 ConvNeXtBlock BasicResidualBlock和 layers [3,4,6]（与你之前一致）
        self.backbone = SeisNetR(ConvNeXtBlock, [3,4,6], in_channels=in_channels)

        # 注意力模块（保持原样）
        self.attention_high = CBAMLayer(1024)
        # ---------------- FFT + FEP（四个分支） ----------------
        # skip
        self.fft_skip1 = FFT_Process(nf=64)
        self.fft_skip2 = FFT_Process(nf=64)
        self.fft_skip3 = FFT_Process(nf=256)

        self.fep_skip1 = FEP_Module(64)
        self.fep_skip2 = FEP_Module(64)
        self.fep_skip3 = FEP_Module(256)

        # 主分支
        self.fft_main = FFT_Process(nf=128)
        self.fep_main = FEP_Module(128)

        # fr 对齐（用于 FFT 交互）
        self.conv_fr_skip1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_fr_skip2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_fr_skip3 = nn.Conv2d(256, 256, 3, 1, 1)
        # ---------------- skip 融合 ----------------
        self.channel_attention_fusion = ChannelAttentionFusion(
            nf_in=64 + 64 + 256,
            nf_out=128
        )

        # ---------------- FGLMSA ----------------
        self.fglmsa = FGLMSA(128)  # skip_fused(128) + aspp(128)

        # ASPP
        self.aspp = LightASPP(1024, 128, atrous_rates)
        self.post_aspp_attention = CBAMLayer(128)

        # Decoder blocks（保持你原来的 module 顺序）
        self.decoder_stage1 = nn.Sequential(
            SubPixelConvBlock(256,128),  # 上采样 ×2，从 (15,9) -> (30,18)
            #nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.Conv2d(128 + 256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

        )
        self.decoder_stage2 = nn.Sequential(
            SubPixelConvBlock(256,128),  # 上采样 ×2，从 (125-ish?) -> see below
            #nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.Conv2d(128 + 64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        #self.subpixel_up3 = SubPixelConvBlock(in_channels=128, out_channels=128, upscale_factor=2)

        self.decoder_stage3 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, 1)
        )
        # 在 Decoder Stage3 后增加宽度转置卷积
        self.up_width = nn.ConvTranspose2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 2),  # 只改变宽度
            stride=(1, 2),
            padding=0,
            output_padding=(0, 0),
            bias=False
        )

    def forward(self, x, label_dsp_dim=(201, 301)):
        # Backbone
        x, skip1, skip2, skip3 = self.backbone(x)
        # ===== skip1 =====
        fep1 = self.fep_skip1(skip1)
        fr1 = self.conv_fr_skip1(skip1)
        skip1, _, _ = self.fft_skip1(skip1, fr1, fep1)

        # ===== skip2 =====
        fep2 = self.fep_skip2(skip2)
        fr2 = self.conv_fr_skip2(skip2)
        skip2, _, _ = self.fft_skip2(skip2, fr2, fep2)

        # ===== skip3 =====
        fep3 = self.fep_skip3(skip3)
        fr3 = self.conv_fr_skip3(skip3)
        skip3, _, _ = self.fft_skip3(skip3, fr3, fep3)
        skip1_up = F.interpolate(skip1, size=skip3.shape[2:], mode='bilinear', align_corners=False)
        skip2_up = F.interpolate(skip2, size=skip3.shape[2:], mode='bilinear', align_corners=False)

        skip_cat = torch.cat([skip1_up, skip2_up, skip3], dim=1)  # [B,384,H,W]
        skip_fused = self.channel_attention_fusion(skip_cat)  # [B,128,H,W]
        # Attention
        x = self.attention_high(x)

        # ASPP + Mellin
        x = self.aspp(x)                 # [B,128,6,9]
        x = self.fglmsa(x)
        #x = self.mellin_conv(x)          # [B,128,6,9]
        x = F.interpolate(x, size=skip_fused.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip_fused, x], dim=1)  # [B,256,H,W]

        # ---------- Decoder Stage 1 ----------
        x = self.decoder_stage1[0](x)
        x = F.interpolate(x, size=skip3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip3], dim=1)
        x = self.decoder_stage1[1:](x)

        # ---------- Decoder Stage 2 ----------
        x = self.decoder_stage2[0](x)
        x = F.interpolate(x, size=skip2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip2], dim=1)
        x = self.decoder_stage2[1:](x)

        # ---------- Decoder Stage 3 ----------
        x = F.interpolate(x, size=skip1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip1], dim=1)
        x = self.decoder_stage3(x)


        # --------- 只上采样宽度，用转置卷积 ---------
        x = self.up_width(x)  # [B,1,200,302]


        # 最终插值到 label_dsp_dim
        if x.shape[2:] != label_dsp_dim:
            x = F.interpolate(x, size=label_dsp_dim, mode='bilinear', align_corners=False)

        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    def _edge_map(self, x):
        fine = self._gradient_mag(x)
        coarse = self._gradient_mag(F.avg_pool2d(x, kernel_size=3, stride=1, padding=1))

        fine = fine / fine.amax(dim=(2, 3), keepdim=True).clamp_min(self.eps)
        coarse = coarse / coarse.amax(dim=(2, 3), keepdim=True).clamp_min(self.eps)

        edge = (1 - self.coarse_weight) * fine + self.coarse_weight * coarse
        return edge

    def forward(self, pred, target):
        err = torch.abs(pred - target)  # L1 误差
        edge = self._edge_map(target)
        w = 1.0 + self.lam * edge
        return (w * err).sum() / (w.sum() + self.eps)


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

        loss_edge = self.edge_loss(pred, target)   # 边界
        loss_inside = self.sp_loss(pred, target)   # 内部 (L2+SSIM)

        loss = balance * loss_edge + (1 - balance) * loss_inside
        return loss, balance


############################################
#                Main Function             #
############################################
if __name__ == '__main__':
    # 生成一个测试输入 (batch_size=2, 5通道, 400x310)
    x = torch.zeros((5, 29, 400, 301))
    model = F3SEG(in_channels=29, n_classes=1)
    label_dsp_dim = [201, 301]
    # 计算输出
    output = model(x, label_dsp_dim)
    print(f"Output shape: {output.size()}")  # 预期: (2, 1, 201, 301)