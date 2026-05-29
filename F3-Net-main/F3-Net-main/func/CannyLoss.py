import torch
import torch.nn as nn
import torch.nn.functional as F

# 拉普拉斯卷积核
LAPLACIAN_KERNEL = torch.tensor([[0., 1., 0.],
                                 [1., -4., 1.],
                                 [0., 1., 0.]]).unsqueeze(0).unsqueeze(0)

class LaplacianLayer(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.register_buffer('kernel', LAPLACIAN_KERNEL.to(device))
    def forward(self, x):
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        x_pad = F.pad(x, (1,1,1,1), mode='reflect')
        return F.conv2d(x_pad, self.kernel)

class SobelEdge(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        gx = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]])
        gy = torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]])
        self.register_buffer('gx', gx.unsqueeze(0).unsqueeze(0).to(device))
        self.register_buffer('gy', gy.unsqueeze(0).unsqueeze(0).to(device))
    def forward(self, x):
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        x_pad = F.pad(x, (1,1,1,1), mode='reflect')
        grad_x = F.conv2d(x_pad, self.gx)
        grad_y = F.conv2d(x_pad, self.gy)
        mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        # 用 sigmoid 近似 Canny 的二值化
        return torch.sigmoid((mag - 0.1) / 0.02)

class CannyLaplaceLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, device='cpu'):
        super().__init__()
        self.alpha = alpha  # Canny 权重
        self.beta = beta    # MSE 权重
        self.gamma = gamma  # Laplace 权重
        self.sobel = SobelEdge(device=device)
        self.laplace = LaplacianLayer(device=device)

    def forward(self, pred, target):
        # 像素 MSE
        loss_mse = F.mse_loss(pred, target)
        # 边缘损失（近似 Canny）
        pred_edge = self.sobel(pred)
        target_edge = self.sobel(target).detach()
        loss_canny = F.mse_loss(pred_edge, target_edge)
        # Laplace 损失
        pred_lap = self.laplace(pred)
        target_lap = self.laplace(target).detach()
        loss_lap = F.mse_loss(pred_lap, target_lap)

        total = self.beta * loss_mse + self.alpha * loss_canny + self.gamma * loss_lap
        return total
