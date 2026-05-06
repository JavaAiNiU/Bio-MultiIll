import os
import wave
import torch
import torch.nn as nn
import numpy as np

# 定义了一个离散小波变换（Discrete Wavelet Transform, DWT）和逆离散小波变换（Inverse DWT）的操作，它们用于对图像进行小波变换和逆变换
# 小波变换是一种在信号和图像处理中常用的技术，用于分析信号和图像的不同频率成分
def dwt_init(x):
    """
    将输入的图像张量 x 分解为四个子图像,分别代表小波变换的低频部分(LL)、水平高频部分(HL)、垂直高频部分(LH)、和对角高频部分(HH)。
    这些子图像经过适当的加权和相加操作后，返回一个包含这四个部分的张量
    """
    # x01 和 x02 是 x 的两个子图像，它们分别包含了 x 中的偶数行和奇数行
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    # x1 包含了 x01 中的偶数列，而 x2 包含了 x02 中的偶数列
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    # x3 包含了 x01 中的奇数列，而 x4 包含了 x02 中的奇数列
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    # 低频部分
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    # 函数返回一个张量，其中包含了上述四个部分，它们在深度维度上连接在一起
    """
    如果x_LL、x_HL、x_LH、x_HH都是形状为(batch_size, num_channels, height, width)的张量，
    那么使用torch.cat((x_LL, x_HL, x_LH, x_HH), 1)将它们连接在一起，
    得到一个形状为(batch_size, 4 * num_channels, height, width)的张量。
    """
    return x_LL, x_HL, x_LH, x_HH
def iwt_init(x):
    """
    用于执行逆离散小波变换，将四个子图像合并还原成原始图像。
    它接受一个包含四个小波变换部分的输入张量，然后执行逆变换操作，返回还原后的原始图像。
    """
    r = 2
    # 
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r**2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)
    # 将四个子图像的信息合并还原成原始图像h
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

class DWT(nn.Module):
    """
    离散小波变换的 PyTorch 模块，它继承自 nn.Module。在其 forward 方法中，它调用了 dwt_init 函数执行小波变换操作，并返回变换后的图像
    """
    def __init__(self):
        super(DWT, self).__init__()
        # 该模块的参数不会进行梯度计算，因为小波变换操作是固定的
        self.requires_grad = False

    def forward(self, x):
        # 执行离散小波变换操作，并将变换后的图像作为结果返回
        return dwt_init(x)

class IWT(nn.Module):
    """执行逆离散小波变换：执行逆变换操作，并返回还原后的图像"""
    def __init__(self):
        super(IWT, self).__init__()
        # 该模块的参数不会进行梯度计算，因为小波变换操作是固定的
        self.requires_grad = False

    def forward(self, x):
        # 执行逆离散小波变换操作，将还原后的图像作为结果返回
        return iwt_init(x)
    
# 计算欧氏距离
def euclidean_distance(x1, x2):
    
    criterion = torch.nn.L1Loss()
    # return torch.sqrt(torch.sum((x1 - x2) ** 2))
    return criterion(x1,x2)
    
    # return torch.sqrt(torch.sum((x1 - x2) ** 2))
    # return torch.sum(torch.abs(x1 - x2))

# 计算LL、LH、HL、HH之间的差距
def compute_wavelet_difference(output,truth):
    dwt = DWT()
    # iwt = IWT()
    out_LL,out_LH,out_HL,out_HH = dwt(output)
    ground_LL,ground_LH,ground_HL,ground_HH = dwt(truth)
    # 计算LL、LH、HL、HH
    ll_diff = euclidean_distance(out_LL, ground_LL)
    lh_diff = euclidean_distance(out_LH, ground_LH)
    hl_diff = euclidean_distance(out_HL, ground_HL)
    hh_diff = euclidean_distance(out_HH, ground_HH)
    total = ll_diff + lh_diff + hl_diff + hh_diff
    return total

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.criterion = torch.nn.L1Loss()
        self.dwt = DWT()
    def forward(self, img1, img2):

        
        # loss = compute_wavelet_difference(img1,img2)

        ll_diff = self.criterion(out_LL, ground_LL)
        lh_diff = self.criterion(out_LH, ground_LH)
        hl_diff = self.criterion(out_HL, ground_HL)
        hh_diff = self.criterion(out_HH, ground_HH)
        return loss

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    dwt = DWT()
    iwt = IWT()
    truth = np.ones((1,4,512,512),dtype=np.float32)
    truth = torch.tensor(truth, dtype=torch.float32, device='cuda')
    output = np.ones((1,4,512,512),dtype=np.float32)
    output = torch.tensor(output, dtype=torch.float32, device='cuda')
    # print(f"input的大小为{input.shape},类型为{type(input)}") # input的大小为torch.Size([1, 4, 512, 512]),类型为<class 'torch.Tensor'>
    # print(f"output的大小为{output.shape},类型为{type(output)}") # output的大小为torch.Size([1, 4, 512, 512]),类型为<class 'torch.Tensor'>
    
    wave_loss = compute_wavelet_difference(output,truth)
    print(wave_loss.item())
    
    # out_LL,out_LH,out_HL,out_HH = dwt(output)
    # ground_LL,ground_LH,ground_HL,ground_HH = dwt(truth)
    # print(f"out_LL的大小为{out_LL.shape},类型为{type(out_LL)}")
    # print(f"out_LH的大小为{out_LH.shape},类型为{type(out_LH)}")
    # print(f"out_HL的大小为{out_HL.shape},类型为{type(out_HL)}")
    # print(f"out_HH的大小为{out_HH.shape},类型为{type(out_HH)}")
    
    # print(f"ground_LL的大小为{ground_LL.shape},类型为{type(ground_LL)}")
    # print(f"ground_LH的大小为{ground_LH.shape},类型为{type(ground_LH)}")
    # print(f"ground_HL的大小为{ground_HL.shape},类型为{type(ground_HL)}")
    # print(f"ground_HH的大小为{ground_HH.shape},类型为{type(ground_HH)}")
    