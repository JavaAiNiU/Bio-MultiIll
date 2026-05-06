import os

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
def apply_wb(org_img,pred,pred_type= "illumination"):
    """
    By using pred tensor (illumination map or uv),
    apply wb into original image (3-channel RGB image).
    """

        
    ones_pred = torch.ones_like(pred[:, :1, :, :])
    
    pred = torch.cat([pred[:, :1, :, :], ones_pred, pred[:, 1:, :, :]], dim=1)

    pred = torch.clamp(torch.abs(pred), min=1e-6)

    pred_rgb = torch.zeros_like(org_img) # b,c,h,w

    if pred_type == "illumination":
        pred_rgb[:,1,:,:] = org_img[:,1,:,:]
        pred_rgb[:,0,:,:] = org_img[:,0,:,:] / (pred[:,0,:,:]+1e-8)    # R_wb = R / illum_R
        pred_rgb[:,2,:,:] = org_img[:,2,:,:] / (pred[:,2,:,:]+1e-8)    # B_wb = B / illum_B
    elif pred_type == "uv":
        pred_rgb[:,1,:,:] = org_img[:,1,:,:]
        pred_rgb[:,0,:,:] = org_img[:,1,:,:] * torch.exp(pred[:,0,:,:])   # R = G * (R/G)
        pred_rgb[:,2,:,:] = org_img[:,1,:,:] * torch.exp(pred[:,1,:,:])   # B = G * (B/G)
    
    return pred_rgb


def calculate_angular_error(pred, gt, tensor_type, camera=None, mask=None):
    """
    计算 Batch 中每一张图片的平均角度误差。
    Returns:
        torch.Tensor: shape (B,), 包含该 batch 中每张图的平均 AE。
    """
    # -----------------------------------------------------------
    
    # -----------------------------------------------------------
    if tensor_type == "rgb":
        if camera == 'galaxy':
            pred = torch.clamp(pred, 0, 1023)
            gt = torch.clamp(gt, 0, 1023)
        elif camera == 'sony' or camera == 'nikon':
            pred = torch.clamp(pred, 0, 16383)
            gt = torch.clamp(gt, 0, 16383)

    # -----------------------------------------------------------
    
    # -----------------------------------------------------------
    
    ones_pred = torch.ones_like(pred[:, :1, :, :])
    pred = torch.cat([pred[:, :1, :, :], ones_pred, pred[:, 1:, :, :]], dim=1)

    ones_gt = torch.ones_like(gt[:, :1, :, :])
    gt = torch.cat([gt[:, :1, :, :], ones_gt, gt[:, 1:, :, :]], dim=1)

    # -----------------------------------------------------------
    
    # -----------------------------------------------------------
    
    cos_similarity = F.cosine_similarity(pred + 1e-4, gt + 1e-4, dim=1)
    cos_similarity = torch.clamp(cos_similarity, -1.0, 1.0)
    rad = torch.acos(cos_similarity)
    ang_error_map = torch.rad2deg(rad) # [B, H, W]

    # -----------------------------------------------------------
    
    # -----------------------------------------------------------
    if mask is not None:
        # mask shape: [B, 1, H, W] -> squeeze -> [B, H, W]
        mask = mask.squeeze(1)
        
        
        
        
        
        masked_error = ang_error_map * mask 
        
        
        error_sum = masked_error.sum(dim=(1, 2))
        pixel_count = mask.sum(dim=(1, 2))
        
        
        mean_angular_error = error_sum / (pixel_count + 1e-8)
        
    else:
        
        mean_angular_error = ang_error_map.mean(dim=(1, 2))

    
    return mean_angular_error


def rgb2uvl(img_rgb):
        epsilon = 1e-8
        img_uvl = np.zeros_like(img_rgb, dtype='float32')
        img_uvl[:,:,2] = np.log(img_rgb[:,:,1] + epsilon)
        img_uvl[:,:,0] = np.log(img_rgb[:,:,0] + epsilon) - img_uvl[:,:,2]
        img_uvl[:,:,1] = np.log(img_rgb[:,:,2] + epsilon) - img_uvl[:,:,2]

        return img_uvl

def plot_illum(pred_map=None,gt_map=None):
    fig = plt.figure()
    if pred_map is not None:
        plt.plot(pred_map[:,0],pred_map[:,1],'ro')
    if gt_map is not None:
        plt.plot(gt_map[:,0],gt_map[:,1],'bx')

    minx,miny = min(gt_map[:,0]),min(gt_map[:,1])
    maxx,maxy = max(gt_map[:,0]),max(gt_map[:,1])
    lenx = (maxx-minx)/2
    leny = (maxy-miny)/2
    add_len = max(lenx,leny) + 0.3

    center_x = (maxx+minx)/2
    center_y = (maxy+miny)/2

    plt.xlim(center_x-add_len,center_x+add_len)
    plt.ylim(center_y-add_len,center_y+add_len)

    # make square
    plt.gca().set_aspect('equal', adjustable='box')

    plt.close()

    fig.canvas.draw()

    return np.array(fig.canvas.renderer._renderer)

def mix_chroma(mixmap,chroma_list,illum_count):
    ret = np.stack((np.zeros_like(mixmap[:,:,0],dtype=float),)*3, axis=2)
    for i in range(len(illum_count)):
        illum_idx = int(illum_count[i])-1
        mixmap_3ch = np.stack((mixmap[:,:,i],)*3, axis=2)
        ret += (mixmap_3ch * [[chroma_list[illum_idx]]])
    
    return ret

def process_files(directory_path):
    """
    检查指定目录中的.pth文件，并从文件名中提取loss和轮数信息。
    
    参数:
    directory_path (str): 目录路径
    
    返回:
    list: 包含每个文件的loss、文件名和epoch的列表
    """
    
    files = [f for f in os.listdir(directory_path) if f.endswith('.pth')]

    
    file_info_list = []

    
    for file_name in files:
        
        parts = file_name.split('_')
        if len(parts) >= 3 and parts[-1].endswith('.pth'):
            loss = parts[-2]
            epoch = parts[-1].split('.')[0]

            
            file_info_list.append({
                '文件名': file_name,
                'Loss': loss,
                'epoch': epoch
            })

    
    if not file_info_list:
        print("目录中没有符合条件的.pth文件。")

    return file_info_list

if __name__ == '__main__':
    
    directory_path = "last_model/"
    file_info_list = process_files(directory_path)
    model_name = file_info_list['文件名']
    model_loss = file_info_list['Loss']
    model_epoch = file_info_list['epoch']
    
   
    print(f"文件名: {model_name}")
    print(f"Loss: {model_loss}")
    print(f"epoch: {model_epoch}")
    print("-" * 50)

