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
        
        # -----------------------------------------------------------
        
        
        ones_pred = torch.ones_like(pred[:, :1, :, :])
        
        pred_full = torch.cat([pred[:, :1, :, :], ones_pred, pred[:, 1:, :, :]], dim=1)

        
        ones_gt = torch.ones_like(gt[:, :1, :, :])
        gt_full = torch.cat([gt[:, :1, :, :], ones_gt, gt[:, 1:, :, :]], dim=1)

        # -----------------------------------------------------------
        
        # -----------------------------------------------------------
        
        
        cos_similarity = F.cosine_similarity(pred_full + self.eps, gt_full + self.eps, dim=1)

        
        cos_similarity = torch.clamp(cos_similarity, -1.0 + self.eps, 1.0 - self.eps)

        
        rad = torch.acos(cos_similarity)
        ang_error = torch.rad2deg(rad) # [B, H, W]

        # -----------------------------------------------------------
        
        # -----------------------------------------------------------
        if mask is not None:
            # mask shape [B, 1, H, W] -> squeeze -> [B, H, W]
            mask_squeezed = mask.squeeze(1)
            
            
            
            valid_errors = ang_error[mask_squeezed != 0]
            
            if valid_errors.numel() > 0:
                loss = valid_errors.mean()
            else:
                
                loss = torch.tensor(0.0, device=pred.device, requires_grad=True)
        else:
            loss = ang_error.mean()

        return loss