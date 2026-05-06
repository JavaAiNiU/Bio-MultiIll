import torch
import torch.nn as nn
import torch.nn.functional as F

class AngularErrorLoss(nn.Module):
    """
    针对 2通道 (R, B) 输入的光照角度误差损失。
    会自动补全 G=1 通道后计算 RGB 向量间的角度误差。
    """
    def __init__(self, eps=1e-7):
        super(AngularErrorLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, gt, mask=None):
        """
        Args:
            pred (torch.Tensor): 预测光照图 (R, B), shape [B, 2, H, W]
            gt (torch.Tensor): 真值光照图 (R, B), shape [B, 2, H, W]
            mask (torch.Tensor, optional): 有效区域掩码, shape [B, 1, H, W]
        """
        # -----------------------------------------------------------
        # 1. 重建 3通道光照图 (R, G=1, B)
        # -----------------------------------------------------------
        # 处理 Pred
        # pred[:, :1, ...] 是 R通道, pred[:, 1:, ...] 是 B通道
        ones_pred = torch.ones_like(pred[:, :1, :, :])
        # 注意：这里千万不能加 .detach()，否则无法训练
        pred_full = torch.cat([pred[:, :1, :, :], ones_pred, pred[:, 1:, :, :]], dim=1)

        # 处理 GT
        ones_gt = torch.ones_like(gt[:, :1, :, :])
        gt_full = torch.cat([gt[:, :1, :, :], ones_gt, gt[:, 1:, :, :]], dim=1)

        # -----------------------------------------------------------
        # 2. 计算角度误差 (基于 F.cosine_similarity)
        # -----------------------------------------------------------
        # F.cosine_similarity 会自动在 dim=1 上计算 dot(A,B)/(|A|*|B|)
        # 输出 shape 为 [B, H, W]
        cos_similarity = F.cosine_similarity(pred_full + self.eps, gt_full + self.eps, dim=1)

        # 数值截断，防止 acos 输入超出 [-1, 1] 导致 NaN
        cos_similarity = torch.clamp(cos_similarity, -1.0 + self.eps, 1.0 - self.eps)

        # 计算弧度并转为角度
        rad = torch.acos(cos_similarity)
        ang_error = torch.rad2deg(rad) # [B, H, W]

        # -----------------------------------------------------------
        # 3. 应用 Mask 计算平均值
        # -----------------------------------------------------------
        if mask is not None:
            # mask shape [B, 1, H, W] -> squeeze -> [B, H, W]
            mask_squeezed = mask.squeeze(1)
            
            # 使用 mask != 0 筛选有效像素
            # valid_errors 变为 1D Tensor
            valid_errors = ang_error[mask_squeezed != 0]
            
            if valid_errors.numel() > 0:
                loss = valid_errors.mean()
            else:
                # 避免空 mask 导致除以 0 错误
                loss = torch.tensor(0.0, device=pred.device, requires_grad=True)
        else:
            loss = ang_error.mean()

        return loss