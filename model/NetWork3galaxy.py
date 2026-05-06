from turtle import forward
import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
# from einops import rearrange
from timm.layers import DropPath, to_2tuple, trunc_normal_
import math


# 激活函数
class LeakyReLU(nn.Module):
    def __init__(self):
        super(LeakyReLU, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        return self.lrelu(x)
    
class MuitlConv(nn.Module):
    """
    保持输入输出通道数一致、特征维度一致的卷积模块
    卷积核大小通过初始化参数传入，激活函数使用LeakyReLU
    """
    def __init__(self, kernel_size, in_channels):
        super(MuitlConv, self).__init__()
        # 计算padding确保输出空间维度与输入一致（stride=1时）
        padding = kernel_size // 2
        # 定义卷积层：输入输出通道一致，步幅为1，padding保证维度不变
        self.stack = nn.Sequential( nn.Conv2d(in_channels, in_channels, kernel_size, 1, padding), LeakyReLU(),
                                    nn.Conv2d(in_channels, in_channels, kernel_size, 1, padding), LeakyReLU())

    def forward(self, x):
        """
        前向传播：卷积 -> LeakyReLU
        Args:
            x: 输入特征，形状为[B, C, H, W]
        Returns:
            输出特征，形状为[B, C, H, W]（与输入维度完全一致）
        """
        x = self.stack(x)
        # x = self.activation(x)
        return x
    
#通道注意力
class CAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CAM, self).__init__()
        self.F_squeeze_avg = nn.AdaptiveAvgPool2d(1)
        self.F_squeeze_max = nn.AdaptiveMaxPool2d(1)

        self.F_excitation = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            # Sigmoid outside compare to SE
        )
    def forward(self, input):
        B,  C,  H,  W = input.size()
        y_avg = self.F_squeeze_avg( input ).view( B, C )
        y_max = self.F_squeeze_max( input ).view( B, C )

        y_merge = self.F_excitation( y_avg ).view( B, C, 1, 1) + self.F_excitation( y_max ).view( B, C, 1, 1)
        output = F.sigmoid(y_merge) * input
        return output
    
#多尺度对立色计算模块
class MultiKernelDiff(nn.Module):
    def __init__(self, in_channels):
        super(MultiKernelDiff, self).__init__()
        # 实例化不同卷积核的卷积层，大小分别为3,5,7,9,11
        self.conv3_1 = MuitlConv(kernel_size=3, in_channels=in_channels)
        self.conv3_2 = MuitlConv(kernel_size=3, in_channels=in_channels)

        self.conv5_1 = MuitlConv(kernel_size=5, in_channels=in_channels)
        self.conv5_2 = MuitlConv(kernel_size=5, in_channels=in_channels)

        self.conv7_1 = MuitlConv(kernel_size=7, in_channels=in_channels)
        self.conv7_2 = MuitlConv(kernel_size=7, in_channels=in_channels)

        self.conv9_1 = MuitlConv(kernel_size=9, in_channels=in_channels)
        self.conv9_2 = MuitlConv(kernel_size=9, in_channels=in_channels)

        self.conv11_1 = MuitlConv(kernel_size=11, in_channels=in_channels)
        self.conv11_2 = MuitlConv(kernel_size=11, in_channels=in_channels)
        #通道注意力结构
        self.Lcam = CAM(channel=in_channels*3)
        self.Scam = CAM(channel=in_channels*3)

    def forward(self, x, y):

        #小尺度单对立色特征计算
        Soutx3 = self.conv3_1(x)
        Souty3 = self.conv3_2(y)

        Soutx5 = self.conv5_1(x)
        Souty5 = self.conv5_2(y)

        Soutx7 = self.conv7_1(x)
        Souty7 = self.conv7_2(y)

        Sdiff1 = Soutx3 - Souty3 
        Sdiff2 = Soutx5 - Souty5 
        Sdiff3 = Soutx7 - Souty7  

        Sresult = torch.cat([Sdiff1, Sdiff2, Sdiff3], dim=1)#48通道
        Sresult = self.Scam(Sresult)#小尺度单对立色特征,48通道

        #大尺度单对立色特征计算
        Loutx7 = self.conv7_1(x)
        Louty7 = self.conv7_2(y)

        Loutx9 = self.conv9_1(x)
        Louty9 = self.conv9_2(y)

        Loutx11 = self.conv11_1(x)
        Louty11 = self.conv11_2(y)

        Ldiff1 = Loutx7 - Louty7 
        Ldiff2 = Loutx9 - Louty9 
        Ldiff3 = Loutx11 - Louty11  

        Lresult = torch.cat([Ldiff1, Ldiff2, Ldiff3], dim=1)
        Lresult = self.Lcam(Lresult)#大尺度单对立色特征,48通道
        
        SO = torch.cat([Sresult, Lresult], dim=1)  #96通道
        DO = Sresult - Lresult   #48通道
        
        return SO,DO

# 定义空间注意模块
class Spatial_Attention(nn.Module):
    def __init__(self):
        super(Spatial_Attention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x * self.sigmoid(x)
    
# 双层卷积 
class UNetConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UNetConvBlock, self).__init__()
        self.UNetConvBlock = torch.nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
            LeakyReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
            LeakyReLU()
        )
    def forward(self, x):
        return self.UNetConvBlock(x)

class Low_Encoder(nn.Module):
    def __init__(self,in_channels) -> None:
        super().__init__()
        self.conv1 = UNetConvBlock(in_channels, 64)  # We have 4 Channel (R,G,B,G)- Bayer Pattern Input
        self.conv2 = UNetConvBlock(64, 128)
        self.conv3 = UNetConvBlock(128, 256)
        # self.conv4 = UNetConvBlock(128, 256)
        
    def forward(self,x):
        L1 = self.conv1(x)  
        # print(f"conv1的大小为{conv1.shape},而它的类型为{type(conv1)}") # conv1的大小为torch.Size([1, 32, 256, 256]),而它的类型为<class 'torch.Tensor'>
        pool1 = F.max_pool2d(L1,kernel_size=2)  
        # print(f"H1的大小为{H1.shape},而它的类型为{type(H1)}") # H1的大小为torch.Size([1, 32, 128, 128]),而它的类型为<class 'torch.Tensor'>
        
        L2 = self.conv2(pool1)  
        # print(f"conv2的大小为{conv2.shape},而它的类型为{type(conv2)}") # conv2的大小为torch.Size([1, 64, 128, 128]),而它的类型为<class 'torch.Tensor'>
        pool2 = F.max_pool2d(L2,kernel_size=2)  
        # print(f"pool2的大小为{pool2.shape},而它的类型为{type(pool2)}") # pool2的大小为torch.Size([1, 64, 64, 64]),而它的类型为<class 'torch.Tensor'>
        
        L3 = self.conv3(pool2)  
        # print(f"conv3的大小为{conv3.shape},而它的类型为{type(conv1)}") # conv3的大小为torch.Size([1, 128, 64, 64]),而它的类型为<class 'torch.Tensor'>
        L4 = F.max_pool2d(L3,kernel_size=2)  
        # print(f"H3的大小为{H3.shape},而它的类型为{type(H3)}") # H3的大小为torch.Size([1, 128, 32, 32]),而它的类型为<class 'torch.Tensor'>
        
        # L4 = self.conv4(pool3)  
        # # print(f"conv4的大小为{conv4.shape},而它的类型为{type(conv4)}") # conv4的大小为torch.Size([1, 256, 32, 32]),而它的类型为<class 'torch.Tensor'>
        
        # poolL = F.max_pool2d(L4,kernel_size=2)  
        # # print(f"poolL的大小为{poolL.shape},而它的类型为{type(poolL)}") # poolL的大小为torch.Size([1, 256, 16, 16]),而它的类型为<class 'torch.Tensor'>
        return L1,L2,L3,L4


class UNet(nn.Module):  # 类名规范首字母大写（可选，增强可读性）
    def __init__(self, in_channels=96):  # 1. 默认输入通道设为96（匹配DO的C=96）
        super(UNet, self).__init__()
        self.in_channels = in_channels  # 目标输出通道=输入通道=96
        # 2. 编码器：输入96通道→逐步下采样提特征（通道数64→128→256）
        self.low_Encoder = Low_Encoder(in_channels)  # 此时Low_Encoder输入为96通道
        
        # -------------------------- 上采样模块（通道逻辑不变，因编码器输出通道固定） --------------------------
        # 上采样1：处理L4（256通道）→与L3（256通道）拼接→512→256
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),  # 上采样后尺寸翻倍，通道保持256
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),  # 1x1卷积统一通道（避免噪声）
            LeakyReLU(),
        )
        self.conv_up1 = UNetConvBlock(512, 256)  # 256（上采样）+256（L3）=512→输出256
        
        # 上采样2：处理256通道→与L2（128通道）拼接→256→128
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 256→128（匹配L2通道）
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            LeakyReLU(),
        )
        self.conv_up2 = UNetConvBlock(256, 128)  # 128（上采样）+128（L2）=256→输出128
        
        # 上采样3：处理128通道→与L1（64通道）拼接→128→128（拼接后通道固定为128）
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 128→64（匹配L1通道）
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            LeakyReLU(),
        )
        self.conv_up3 = UNetConvBlock(128, 128)  # 64（上采样）+64（L1）=128→输出128（关键：固定为128，为后续降维做准备）
        

        # -------------------------- 动态生成convlast1（128→in_channels） --------------------------
        # 核心：通过私有方法_build_convlast1动态构建降维序列，保持原结构形式
        self.convlast1 = self._build_convlast1(
            start_channel=128,  # 输入固定为128通道（第三次拼接后输出）
            target_channel=in_channels  # 输出匹配输入通道数
        )

    def _build_convlast1(self, start_channel, target_channel):
        """
        私有方法：动态构建convlast1模块
        结构规则：每步保持「3x3卷积（保通道）→LeakyReLU→1x1卷积（降维）→LeakyReLU」
        降维逻辑：从start_channel开始，每次优先除以2，最后一步调整至target_channel
        """
        layers = []
        current_ch = start_channel  # 当前通道数（初始为128）

        # 循环生成降维步骤，直到通道数降至target_channel
        while current_ch > target_channel:
            # 计算下一步通道数：优先//2（保证降维效率），若//2小于目标则直接用目标
            next_ch = current_ch // 2 if (current_ch // 2) >= target_channel else target_channel
            
            # 添加当前步骤的4层结构（与原convlast1形式完全一致）
            layers.extend([
                # 3x3卷积：保持当前通道数，提取空间特征
                nn.Conv2d(current_ch, current_ch, kernel_size=3, stride=1, padding=1),
                LeakyReLU(),  # 激活函数
                # 1x1卷积：将当前通道数降维至next_ch（实现通道压缩）
                nn.Conv2d(current_ch, next_ch, kernel_size=1, stride=1, padding=0),
                LeakyReLU()  # 激活函数
            ])
            
            current_ch = next_ch  # 更新当前通道数，进入下一轮循环

        # 构建并返回Sequential序列
        return nn.Sequential(*layers)

    def forward(self, x):
        """前向传播：编码器→三次上采样拼接→动态降维→输出"""
        # 1. 编码器输出：L1(64, H, W)、L2(128, H/2, W/2)、L3(256, H/4, W/4)、L4(256, H/8, W/8)
        L1, L2, L3, L4 = self.low_Encoder(x)

        # 2. 第一次上采样与拼接
        x = self.up1(L4)  # L4上采样→(256, H/4, W/4)（与L3尺寸一致）
        x = torch.cat([x, L3], dim=1)  # 拼接→(512, H/4, W/4)
        x = self.conv_up1(x)  # 卷积→(256, H/4, W/4)

        # 3. 第二次上采样与拼接
        x = self.up2(x)  # 上采样→(128, H/2, W/2)（与L2尺寸一致）
        x = torch.cat([x, L2], dim=1)  # 拼接→(256, H/2, W/2)
        x = self.conv_up2(x)  # 卷积→(128, H/2, W/2)

        # 4. 第三次上采样与拼接
        x = self.up3(x)  # 上采样→(64, H, W)（与L1尺寸一致）
        x = torch.cat([x, L1], dim=1)  # 拼接→(128, H, W)（匹配convlast1输入）
        x = self.conv_up3(x)  # 输出：B×128×H×W

        # 5. 动态降维：128通道→in_channels通道并return
        return self.convlast1(x)

        # return x_RAW

class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=3, stride=1, in_chans=3, embed_dim=48):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                            #   padding=(patch_size[0] // 2, patch_size[1] // 2))
        
        mid_dim = (in_chans + embed_dim) // 2 
        self.proj = nn.Sequential(
            # 第一步：低维→中间维（保留原 kernel/stride/padding，确保空间尺寸不变）
            nn.Conv2d(
                in_channels=in_chans, 
                out_channels=mid_dim, 
                kernel_size=patch_size, 
                stride=stride,
                padding=(patch_size[0] // 2, patch_size[1] // 2)
            ),
            LeakyReLU(),  # 中间激活增强非线性（与项目中其他模块激活函数保持一致）
            # 第二步：中间维→目标维（同样保留原空间参数，确保输出尺寸与原代码完全一致）
            nn.Conv2d(
                in_channels=mid_dim, 
                out_channels=embed_dim, 
                kernel_size=patch_size, 
                stride=stride,
                padding=(patch_size[0] // 2, patch_size[1] // 2)
            )
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m): 
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        if self.patch_size[0] == 7:
            x = self.proj(x)
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
        else:
            # x = x.permute(0, 3, 1, 2)
            x = self.proj(x)
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
        return x  
    
class Cross_Attention(nn.Module):
    def __init__(self, key_channels, value_channels, head_count=1):
        super().__init__()
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        self.reprojection = nn.Conv2d(value_channels, 2 * value_channels, 1)
        self.norm = nn.LayerNorm(2 * value_channels)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.25]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.25]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.25]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.25]), requires_grad=True)
        # self.attn5 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    # x2 should be higher-level representation than x1
    #比较的值在前，主干在后
    def forward(self, x1, x2, H, W):
        B, N, D = x1.size()  # (Batch, Tokens, Embedding dim)
        # Re-arrange into a (Batch, Embedding dim, Tokens)
        # 代码写反了,V实际是Q,Q实际是V
        keys = x1.transpose(1, 2)
        queries = x2.transpose(1, 2)
        values = x1.transpose(1, 2)
        
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            
            key = F.softmax(keys[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels : (i + 1) * head_value_channels, :]

            context = key @ value.transpose(1, 2)  # dk*dv

            # print("key:",key.shape)
            # print("value.transpose(1, 2):",value.transpose(1, 2).shape)
            # print("context:",context.shape)
            
            mask1 = torch.zeros(B, D, D, device=x1.device, requires_grad=False)
            mask2 = torch.zeros(B, D, D, device=x1.device, requires_grad=False)
            mask3 = torch.zeros(B, D, D, device=x1.device, requires_grad=False)
            mask4 = torch.zeros(B, D, D, device=x1.device, requires_grad=False)

            index = torch.topk(context, k=int(D * 1 / 2), dim=-1, largest=True)[1]
            mask1.scatter_(-1, index, 1.)
            attn1 = torch.where(mask1 > 0, context, torch.full_like(context, float('-inf')))

            index = torch.topk(context, k=int(D * 2 / 3), dim=-1, largest=True)[1]
            mask2.scatter_(-1, index, 1.)
            attn2 = torch.where(mask2 > 0, context, torch.full_like(context, float('-inf')))

            index = torch.topk(context, k=int(D * 3 / 4), dim=-1, largest=True)[1]
            mask3.scatter_(-1, index, 1.)
            attn3 = torch.where(mask3 > 0, context, torch.full_like(context, float('-inf')))

            index = torch.topk(context, k=int(D * 4 / 5), dim=-1, largest=True)[1]
            mask4.scatter_(-1, index, 1.)
            attn4 = torch.where(mask4 > 0, context, torch.full_like(context, float('-inf')))

            # attn5 = maskv
            attn1 = attn1.softmax(dim=-1)
            attn2 = attn2.softmax(dim=-1)
            attn3 = attn3.softmax(dim=-1)
            attn4 = attn4.softmax(dim=-1)
            # attn5 = attn5.softmax(dim=-1)
            out1 = (attn1 @ query)
            # print("out1:",out1.shape)
            out2 = (attn2 @ query)
            out3 = (attn3 @ query)
            out4 = (attn4 @ query)
            # out5 = (attn5 @ query)

            attended_value = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4 #+ out5 * self.attn5
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1).reshape(B, D, H, W)
        # print("aggregated_values:",aggregated_values.shape)
        reprojected_value = self.reprojection(aggregated_values).reshape(B, 2 * D, N).permute(0, 2, 1)
        # print("reprojected_value1:",reprojected_value.shape)     
        reprojected_value = self.norm(reprojected_value)
        # print("reprojected_value2:",reprojected_value.shape)     

        return reprojected_value
        
class CrossAttentionBlock(nn.Module):
    """
    Input ->    x1:[B, N, D] - N = H*W
                x2:[B, N, D]
    Output -> y:[B, N, D]
    D is half the size of the concatenated input (x1 from a lower level and x2 from the skip connection)
    """
    def __init__(self, in_dim, key_dim, value_dim,head_count=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = Cross_Attention(key_dim, value_dim, head_count=head_count)
        # self.norm2 = nn.LayerNorm((in_dim*2))
        # self.mlp = MixFFN_skip(int(in_dim*2) , int(in_dim * 4))
        # self.channel_att = SELayer(channel=int(in_dim*2))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        B,H,W,C = x1.shape
        x1 = x1.reshape(B, H*W, C)
        x2 = x2.reshape(B, H*W, C)
        norm_1 = self.norm1(x1)
        norm_2 = self.norm1(x2)
        
        attn = self.attn(norm_1, norm_2, H, W)
        # residual = torch.cat([x1, x2], dim=2)
        # tx = residual + attn
        # tx = attn
        # mx = tx + self.mlp(self.norm2(tx),H,W)
        # cx = mx.view(B, H, W, 2 * C).permute(0, 3, 1, 2)
        attn = attn.view(B, H, W, 2 * C).permute(0, 3, 1, 2)
        # cx = self.channel_att(cx)
        return attn

# 后处理卷积：输入任意N通道，动态降维至固定16通道
class PsotConvBlock(nn.Module):
    def __init__(self, in_channel,out_channel):  # 仅需传入「输入通道数N」，目标输出固定为16
        # 1. 修正继承错误：原继承UNetConvBlock，需改为继承自身PsotConvBlock
        super(PsotConvBlock, self).__init__()
        
        # 2. 动态降维逻辑：分两步过渡（避免一步降维导致特征丢失，梯度更稳定）
        # 中间通道数设计：取「输入通道的1/2」与「32」的最大值（确保过渡平滑，且不小于16的2倍）
        mid_channel = max(in_channel // 2, 32)
        
        # 3. 卷积序列：输入N通道 → 中间通道 → 固定16通道（每步配激活函数）
        self.conv_block = torch.nn.Sequential(
            # 第一步：N通道 → 中间通道（3x3卷积保留空间特征，padding=1保证尺寸不变）
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1),
            LeakyReLU(),
            nn.Conv2d(in_channels=in_channel, out_channels=mid_channel, kernel_size=1, stride=1, padding=0),
            LeakyReLU(),  

            nn.Conv2d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, padding=1),
            LeakyReLU(),  # 激活函数增强非线性
            nn.Conv2d(in_channels=mid_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            LeakyReLU(),  

            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
            LeakyReLU(),  
        )
    
    def forward(self, x):
        # 前向传播：直接通过卷积序列，输出固定16通道
        return self.conv_block(x)
    
# 后处理卷积：输入任意N通道，动态降维至固定16通道
class PsotConvBlock2(nn.Module):
    def __init__(self, in_channel,out_channel):  # 仅需传入「输入通道数N」，目标输出固定为16
        # 1. 修正继承错误：原继承UNetConvBlock，需改为继承自身PsotConvBlock
        super(PsotConvBlock2, self).__init__()

        # 3. 卷积序列：输入N通道 → 中间通道 → 固定16通道（每步配激活函数）
        self.conv_block = torch.nn.Sequential(
            # 第一步：N通道 → 中间通道（3x3卷积保留空间特征，padding=1保证尺寸不变）
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1),
            LeakyReLU(),
            nn.Conv2d(in_channels=in_channel, out_channels=128, kernel_size=1, stride=1, padding=0),
            LeakyReLU(),  

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),
            LeakyReLU(),  

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            LeakyReLU(),  
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0),
            LeakyReLU(),  

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            LeakyReLU(),  
            nn.Conv2d(in_channels=32, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            LeakyReLU(),  
        )
    
    def forward(self, x):
        # 前向传播：直接通过卷积序列，输出固定16通道
        return self.conv_block(x)
    
class PostModule(nn.Module):
    def __init__(self):
    # def __init__(self):
        super(PostModule, self).__init__()

        self.SO = PsotConvBlock(96,16)

        self.cam = CAM(channel=96)

        self.DO = PsotConvBlock2(96,16)

        self.patch_embed_C = OverlapPatchEmbed(patch_size=3, stride=1, in_chans=16, embed_dim=64)#B,H,W,C
        self.patch_embed_IF = OverlapPatchEmbed(patch_size=3, stride=1, in_chans=16, embed_dim=64)#B,H,W,C

        self.fuse = CrossAttentionBlock(64,64,64)

        self.conv_residual = nn.Sequential(nn.Conv2d(128, 96, 1), LeakyReLU(),
                                          nn.Conv2d(96, 96, 1), LeakyReLU(),
                                          nn.Conv2d(96, 64, 1), LeakyReLU(),
                                          nn.Conv2d(64, 32, 1), LeakyReLU(),
                                          nn.Conv2d(32, 16, 1), LeakyReLU(),
        )

        self.g_conv_post = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0),
            LeakyReLU(),
            nn.Conv2d(16, 12, kernel_size=1, stride=1, padding=0),
            LeakyReLU(),
            nn.Conv2d(12, 6, kernel_size=1, stride=1, padding=0),
            LeakyReLU(),
            nn.Conv2d(6, 2, kernel_size=1, stride=1, padding=0),
            LeakyReLU(),
            )
        
        # self.up = nn.PixelShuffle(2)

        self.finish_second = UNetConvBlock(2,2) 
        
    def forward(self,F,SO,V,IF=0):

        SO = self.cam(SO)#先进行通道注意力，区分物体本色和光源颜色 16
        SO = self.SO(F*SO)#在进行反馈优化 16
        V =self.DO(V)# 16
        V_e = self.patch_embed_C(V)#,64
        SO_e = self.patch_embed_C(SO)#,64


        SO_e = self.fuse(V_e,SO_e)#自注意力优化远程色彩信息,128

        #残差拼接
        IF = self.conv_residual(SO_e)#16
        SO = SO+IF#16
        color = self.g_conv_post(SO)

        color = self.finish_second(color)

        return color


class  initModule(nn.Module):
    def __init__(self):
        super(initModule, self).__init__()

        self.second_stagebefore = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3,padding=1),
            LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3,padding=1),
            LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3,padding=1),
            LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            LeakyReLU()
        )

        self.Convg = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            LeakyReLU(),
            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0),  # 1x1卷积降维，尺寸不变
            LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0),
            LeakyReLU()
        )

        self.MSO_RG = MultiKernelDiff(16)
        self.MSO_YB = MultiKernelDiff(16)

        self.Convg2 = nn.Sequential(
            nn.Conv2d(192, 128 , kernel_size=1, stride=1, padding=0),
            LeakyReLU(),
            nn.Conv2d(128, 96, kernel_size=1, stride=1, padding=0),  # 1x1卷积降维，尺寸不变
            LeakyReLU(),
            nn.Conv2d(96, 96, kernel_size=1, stride=1, padding=0),
            LeakyReLU()
        )

        # self.w_r = nn.Parameter(torch.tensor(0.299, dtype=torch.float32))  # r 通道权重
        # self.w_g = nn.Parameter(torch.tensor(0.587, dtype=torch.float32))  # g 通道权重
        # self.w_b = nn.Parameter(torch.tensor(0.114, dtype=torch.float32))  # b 通道权重

    def split_feature_map(self,feature_map, num_groups=4):
        input_channels = feature_map.size(1)
        group_size = input_channels // num_groups
        groups = []
        for i in range(num_groups):
            start_channel = i * group_size
            end_channel = (i + 1) * group_size
            group = feature_map[:, start_channel:end_channel, :, :]
            groups.append(group)
        LL = groups[0]
        LH = groups[1]
        HL = groups[2]
        HH = groups[3]
        return LL,LH,HL,HH
    
    def forward(self,x_raw):

        x = self.second_stagebefore(x_raw) # 3->64
        r,G1,G2,b = self.split_feature_map(x) #64拆分成4个16
        g = torch.cat([G1,G2],dim=1)#
        g = self.Convg(g)#32->16

        SO_RG,DO_RG = self.MSO_RG(r,g) #96,48
        SO_YB,DO_YB = self.MSO_YB(b,(r+g)/2) #96,48

        # W = self.w_r * r + self.w_g * g + self.w_b * b #通道数是16

        return self.Convg2(torch.cat([SO_RG,SO_YB],dim=1)),torch.cat([DO_RG,DO_YB],dim=1)#,W #96,96,16
    

# class FkCalculator(nn.Module):
#     """
#     计算公式 (分子分母同乘 σ_1^2 简化版):
#     F_k = ( (L_{k-1} ⊙ S) + 2μ_F * σ_1^2 * Z_{F, k-1} ) / ( (S ⊙ S) + 2μ_F * σ_1^2 )
    
#     Args:
#         sigma1_init (float): 可学习参数 σ_1 的初始值，默认 1.0
#         mu_f_init (float): 可学习参数 μ_F 的初始值，默认 0.1
#         eps (float): 防止分母过小的微小值，默认 1e-8
#     """
class FkCalculator(nn.Module):
    def __init__(self, sigma1_init: float = 1.0, mu_f_init: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.sigma1 = nn.Parameter(torch.tensor(sigma1_init, dtype=torch.float32), requires_grad=True)
        self.mu_f = nn.Parameter(torch.tensor(mu_f_init, dtype=torch.float32), requires_grad=True)
        self.eps = eps

    def forward(self, k, S, L_prev, Z_f_prev):
        if k <= 0: raise ValueError(f"k must be > 0")

        # 【修复1】: 强制参数非负 (Softplus 平滑且恒正)
        # 物理参数 μ 和 σ 不应为负数，否则分母 (S^2 + coeff) 可能变成 0
        mu_f_safe = F.softplus(self.mu_f)
        sigma1_safe = self.sigma1 # sigma 平方后必为正，这里主要防 mu
        
        coeff = 2 * mu_f_safe * sigma1_safe.pow(2)

        # 【修复2】: 防止 S 过大导致平方溢出 (fp32 上限)
        # 限制 S 的范围在 [-100, 100] 之间，防止 S*S 变成 Inf
        S_safe = torch.clamp(S, min=-100.0, max=100.0) 

        numerator = (L_prev * S_safe) + (coeff * Z_f_prev)

        # 【修复3】: 分母加固
        denominator = (S_safe * S_safe) + coeff + self.eps
        
        return numerator / denominator


class VkMultiConv(nn.Module):
    """
    保持输入输出通道数一致、特征维度一致的卷积模块
    结构: Conv(3x3) -> LeakyReLU -> Conv(3x3) -> LeakyReLU
    """
    def __init__(self, in_channels, kernel_size=3, negative_slope=0.2):
        super(VkMultiConv, self).__init__()
        # 计算padding确保输出空间维度与输入一致（stride=1时）
        padding = kernel_size // 2
        
        self.stack = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding=padding),
            LeakyReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding=padding),
            LeakyReLU()
        )

    def forward(self, x):
        return self.stack(x)


# class VkCalculator(nn.Module):
#     """
#     计算公式:
#     V_k = ( Net(L_k) + 2 * σ_2^2 * μ_V * Z_{V, k-1} ) / ( 1 + 2 * σ_2^2 * μ_V )
    
#     其中 Net(L_k) 代替了原本的 K_L ★ L_k
#     """
#     def __init__(self, channels: int, sigma2_init: float = 1.0, mu_v_init: float = 0.1, eps: float = 1e-8):
#         """
#         Args:
#             channels (int): 输入特征的通道数（用于初始化卷积层）
#             sigma2_init, mu_v_init, eps: 参数同前
#         """
#         super().__init__()
        
#         # --- 核心变化：实例化一个卷积模块来代替 K_L ---
#         self.KL_net = VkMultiConv(in_channels=channels, kernel_size=3)
        
#         # 可学习参数 σ_2
#         self.sigma2 = nn.Parameter(
#             torch.tensor(sigma2_init, dtype=torch.float32),
#             requires_grad=True
#         )
        
#         # 可学习参数 μ_V
#         self.mu_v = nn.Parameter(
#             torch.tensor(mu_v_init, dtype=torch.float32),
#             requires_grad=True
#         )
        
#         self.eps = eps

#     def forward(
#         self, 
#         k: int,                 # 当前迭代步骤
#         Lk: torch.Tensor,       # 输入 L_k [B, C, H, W]
#         Zv_prev: torch.Tensor   # 输入 Z_{V, k-1} [B, C, H, W]
#     ) -> torch.Tensor:
#         """
#         前向传播计算 V_k
#         注意：此时不再需要传入 KL_kernel，因为已经内化为 self.KL_net
#         """
#         # 1. 校验与维度检查
#         if k <= 0:
#             raise ValueError(f"公式仅定义 k>0 的情况")
#         if Lk.shape != Zv_prev.shape:
#             raise ValueError(f"Lk 和 Zv_prev 维度必须一致: {Lk.shape} vs {Zv_prev.shape}")

#         # 2. 使用卷积网络计算第一项 (代替 K_L ★ L_k)
#         # 这一步会自动保持维度不变 [B, C, H, W]
#         conv_term = self.KL_net(Lk)
        
#         # 3. 计算公共系数: coeff = 2 * σ_2^2 * μ_V
#         coeff = 2 * self.sigma2.pow(2) * self.mu_v
        
#         # 4. 计算分子
#         # Term 1: Net(L_k)
#         # Term 2: coeff * Z_{V, k-1}
#         numerator = conv_term + (coeff * Zv_prev)

#         # 5. 计算分母
#         # Term 1: 1
#         # Term 2: coeff
#         denominator = 1 + coeff + self.eps
        
#         # 6. 返回结果
#         return numerator / denominator
class VkCalculator(nn.Module):
    def __init__(self, channels: int, sigma2_init: float = 1.0, mu_v_init: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.KL_net = VkMultiConv(in_channels=channels, kernel_size=3)
        self.sigma2 = nn.Parameter(torch.tensor(sigma2_init, dtype=torch.float32), requires_grad=True)
        self.mu_v = nn.Parameter(torch.tensor(mu_v_init, dtype=torch.float32), requires_grad=True)
        self.eps = eps

    def forward(self, k, Lk, Zv_prev):
        if k <= 0: raise ValueError(f"k must be > 0")
        
        conv_term = self.KL_net(Lk)
        
        # 【修复1】: 强制参数非负
        mu_v_safe = F.softplus(self.mu_v)
        
        coeff = 2 * self.sigma2.pow(2) * mu_v_safe
        
        numerator = conv_term + (coeff * Zv_prev)

        # 【修复2】: 分母加固，防止 1+coeff 意外抵消
        denominator = 1 + coeff + self.eps
        
        return numerator / denominator
    

class LkMultiConv(nn.Module):
    """
    基础卷积模块：保持尺寸不变 (Same Padding)
    结构: Conv(3x3) -> LeakyReLU -> Conv(3x3) -> LeakyReLU
    """
    def __init__(self, in_channels, kernel_size=3, negative_slope=0.2):
        super(LkMultiConv, self).__init__()
        padding = kernel_size // 2
        self.stack = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding=padding),
            LeakyReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding=padding),
            LeakyReLU()
        )

    def forward(self, x):
        return self.stack(x)
    
class DownSampleConv(nn.Module):
    """
    专用下采样卷积模块 (用于 V_{k-1})
    结构: 
    1. Conv(3x3, stride=1): 提取特征，保持尺寸
    2. Conv(3x3, stride=2): 下采样，尺寸减半
    """
    def __init__(self, in_channels, kernel_size=3, negative_slope=0.2):
        super(DownSampleConv, self).__init__()
        padding = kernel_size // 2
        
        self.stack = nn.Sequential(
            # 第一层：特征变换 (H, W) -> (H, W)
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding=padding),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            
            # 第二层：下采样 (H, W) -> (H/2, W/2)
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=2, padding=padding),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        )

    def forward(self, x):
        return self.stack(x)


class MultiConv(nn.Module):
    """
    基础卷积模块：保持尺寸不变 (Same Padding)
    结构: Conv(3x3) -> LeakyReLU -> Conv(3x3) -> LeakyReLU
    """
    def __init__(self, in_channels, kernel_size=3, negative_slope=0.2):
        super(MultiConv, self).__init__()
        padding = kernel_size // 2
        self.stack = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding=padding),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding=padding),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        )

    def forward(self, x):
        return self.stack(x)


class LkCalculator(nn.Module):
    def __init__(self, channels: int, sigma1_init: float = 1.0, sigma2_init: float = 1.0, mu_l_init: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.L_feature_net = MultiConv(channels)
        self.V_down_net = DownSampleConv(channels)
        self.up_restore = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.sigma1 = nn.Parameter(torch.tensor(sigma1_init, dtype=torch.float32), requires_grad=True)
        self.sigma2 = nn.Parameter(torch.tensor(sigma2_init, dtype=torch.float32), requires_grad=True)
        self.mu_l = nn.Parameter(torch.tensor(mu_l_init, dtype=torch.float32), requires_grad=True)
        self.eps = eps

    def forward(self, k, S, Fk, ZL, Lk_prev, Vk_prev):
        if k <= 0: raise ValueError("k must be > 0")

        # L/V 网络部分保持不变...
        feat_L = self.L_feature_net(Lk_prev)
        down_L = F.max_pool2d(feat_L, kernel_size=2)
        down_V = self.V_down_net(Vk_prev)
        residual_low = down_V - down_L
        residual_high = self.up_restore(residual_low)
        
        if residual_high.shape != S.shape:
             residual_high = F.interpolate(residual_high, size=S.shape[2:], mode='bilinear', align_corners=False)

        # === 【核心修复区域】 ===
        
        # 1. 强制 mu_l 非负
        mu_l_safe = F.softplus(self.mu_l)
        coeff_mu = 2 * mu_l_safe * self.sigma1.pow(2)

        # 2. 防止 sigma2 趋近于 0
        sigma2_sq = self.sigma2.pow(2) + 1e-4  # 稍微加大基底到 1e-4 更稳定
        
        sigma_ratio = self.sigma1.pow(2) / sigma2_sq
        
        # 【关键修复】: 降低最大增益上限
        # 这里的 ratio 决定了 residual (V-L) 对结果的影响权重
        # 原来的 10000.0 太大了，意味着一点点误差会被放大一万倍
        # 建议改为 10.0 或 100.0
        sigma_ratio = torch.clamp(sigma_ratio, max=50.0) 

        numerator = (Fk * S) + (coeff_mu * ZL) + (sigma_ratio * residual_high)
        denominator = 1 + coeff_mu + self.eps

        return numerator / denominator

    
# class LkCalculator(nn.Module):
#     """
#     计算公式: L_k = ... UpNet(V_{k-1} - DownNet(L_{k-1})) ...
    
#     修改点:
#     1. L_{k-1} 路径: MultiConv + MaxPool
#     2. V_{k-1} 路径: 使用 DownSampleConv (两层卷积) 替代双线性插值
#     3. UpNet 路径: 反卷积 + 1x1卷积
#     """
#     def __init__(self, channels: int, sigma1_init: float = 1.0, sigma2_init: float = 1.0, mu_l_init: float = 0.1, eps: float = 1e-8):
#         super().__init__()
        
#         # 1. L_{k-1} 的处理网络 (MultiConv + MaxPool)
#         self.L_feature_net = MultiConv(channels)
        
#         # 2. V_{k-1} 的处理网络 (新增: 两层卷积下采样)
#         self.V_down_net = DownSampleConv(channels)
        
#         # 3. 恢复网络 (反卷积 + 1x1)
#         self.up_restore = nn.Sequential(
#             nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2),
#             nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True)
#         )
        
#         # 4. 可学习参数
#         self.sigma1 = nn.Parameter(torch.tensor(sigma1_init, dtype=torch.float32), requires_grad=True)
#         self.sigma2 = nn.Parameter(torch.tensor(sigma2_init, dtype=torch.float32), requires_grad=True)
#         self.mu_l = nn.Parameter(torch.tensor(mu_l_init, dtype=torch.float32), requires_grad=True)
        
#         self.eps = eps

#     def forward(
#         self, 
#         k: int,                 
#         S: torch.Tensor,        
#         Fk: torch.Tensor,       
#         ZL: torch.Tensor,                         
#         Lk_prev: torch.Tensor,  # 高分辨率 [B, C, H, W]
#         Vk_prev: torch.Tensor   # 高分辨率 [B, C, H, W]
#     ) -> torch.Tensor:
        
#         if k <= 0: raise ValueError("k must be > 0")

#         # === 步骤 1: 处理 L_{k-1} ===
#         # 路径: 卷积特征提取 -> 最大池化降采样
#         # Output: [B, C, H/2, W/2]
#         feat_L = self.L_feature_net(Lk_prev)
#         down_L = F.max_pool2d(feat_L, kernel_size=2)
        
#         # === 步骤 2: 处理 V_{k-1} (完全卷积化) ===
#         # 路径: 卷积特征提取 -> 步长卷积降采样
#         # Output: [B, C, H/2, W/2]
#         down_V = self.V_down_net(Vk_prev)

#         # 校验维度 (确保两路网络输出尺寸一致)
#         if down_V.shape != down_L.shape:
#              # 如果出现尺寸不匹配(通常是奇数输入造成的边缘差异)，这里可能需要异常处理
#              raise ValueError(f"L路与V路下采样后尺寸不一致: L={down_L.shape}, V={down_V.shape}")

#         # === 步骤 3: 计算残差并上采样 ===
#         # 此时两者都在低分辨率空间
#         residual_low = down_V - down_L
        
#         # 恢复到高分辨率 [B, C, H, W]
#         residual_high = self.up_restore(residual_low)
        
#         # 最终尺寸安全检查
#         if residual_high.shape != S.shape:
#             # 仅在极端边缘情况下做插值兜底，正常情况下卷积逻辑应保证尺寸匹配
#             residual_high = F.interpolate(residual_high, size=S.shape[2:], mode='bilinear', align_corners=False)

#         # === 步骤 4: 套用公式 ===
#         coeff_mu = 2 * self.mu_l * self.sigma1.pow(2)
#         sigma_ratio = self.sigma1.pow(2) / (self.sigma2.pow(2) + self.eps)

#         numerator = (Fk * S) + (coeff_mu * ZL) + (sigma_ratio * residual_high)
#         denominator = 1 + coeff_mu + self.eps

#         return numerator / denominator


class BasicBlock(nn.Module):
    """标准的 ResNet BasicBlock"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        # 如果维度发生变化（stride!=1 或 输入输出通道不一致），需要对 shortcut 进行投影
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18_Prior(nn.Module):
    """
    专为 Deep Unfolding Network 设计的 ResNet-18 先验网络
    结构：U-Net 风格 (ResNet18 Encoder + Light Decoder)
    输入: [B, 2, H, W] (LGN特征)
    输出: [B, 2, H, W] (去噪后的辅助变量 Z_L)
    """
    def __init__(self, in_channels=2, out_channels=2):
        super(ResNet18_Prior, self).__init__()
        self.in_planes = 64

        # --- Encoder (标准 ResNet-18 结构) ---
        # 1. 初始层：修改了输入通道数 (通常是3，这里改为 in_channels=2)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # 注意：去掉了 MaxPool，为了保留更多空间细节，适合底层视觉任务
        
        # 2. ResNet Layers
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2) # Downsample /2
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2) # Downsample /4
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2) # Downsample /8

        # --- Decoder (上采样恢复尺寸) ---
        # 使用转置卷积或插值+卷积来恢复分辨率
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # --- Output Head ---
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # --- Encoder ---
        out = F.relu(self.bn1(self.conv1(x))) # [B, 64, H, W]
        c1 = out # Skip connection ready
        
        out = self.layer1(out) # [B, 64, H, W]
        out = self.layer2(out) # [B, 128, H/2, W/2]
        out = self.layer3(out) # [B, 256, H/4, W/4]
        out = self.layer4(out) # [B, 512, H/8, W/8]
        
        # --- Decoder ---
        out = self.up3(out)    # [B, 256, H/4, W/4]
        # 这里可以加 Skip Connection: out = out + enc_feat
        
        out = self.up2(out)    # [B, 128, H/2, W/2]
        out = self.up1(out)    # [B, 64, H, W]
        
        # --- Output ---
        # 建议使用残差学习：网络只学习"修正量"
        # Z_L = L + Net(L)
        residual = self.final_conv(out)
        
        return x + residual
    
class DUM(nn.Module):
    def __init__(self):
        super(DUM, self).__init__()
        
        # --- 1. 物理求解器 ---
        self.step_F = FkCalculator()
        self.step_L = LkCalculator(channels=96)
        self.step_V = VkCalculator(channels=96)

        # --- 2. 先验网络 ---
        self.prior_F = Spatial_Attention() 
        self.prior_L = ResNet18_Prior(in_channels=96, out_channels=96)
        self.prior_V = UNet(in_channels=96)

        # 维度对齐
        self.proj_96to1 = nn.Conv2d(96, 1, kernel_size=1, stride=1, padding=0, bias=True)
        
        # Loss 权重
        self.w_F = nn.Parameter(torch.tensor(1.0))
        self.w_L = nn.Parameter(torch.tensor(1.0))
        self.w_V = nn.Parameter(torch.tensor(1.0))

        self.norm_L = nn.GroupNorm(num_groups=16, num_channels=96)
        self.norm_V = nn.GroupNorm(num_groups=16, num_channels=96)

    def forward(self, SO, DO_init):
        # --- 初始化 ---
        S = SO
        L = S.clone()
        V = DO_init.clone()
        F_val = torch.ones_like(S)
        
        Z_L = L.clone()
        Z_V = V.clone()
        Z_F = F_val.clone()
        
        loss_aux = 0.0

        # --- HQS 迭代循环 (3次) ---
        for i in range(3):
            k = i + 1 
            
            # === Step 1: 更新 F ===
            F_val = self.step_F(k, S, L, Z_F)
            # F 是光照图，物理意义必须非负，且通常在 0~10 之间
            # 这里不用 Norm，而是用物理约束
            F_val = torch.clamp(F_val, min=0.01, max=10.0) 
            Z_F = self.prior_F(F_val)
            
            # === Step 2: 更新 L ===
            L = self.step_L(k, S, F_val, Z_L, L, V)
            
            # 【核心修改 A】：主动归一化 L
            # 不让它有机会累积到 500，每一步都重置分布
            L = self.norm_L(L) 
            
            # 这里的 clamp 可以放宽，或者作为最后的保险
            L = torch.clamp(L, min=-50.0, max=50.0)
            
            Z_L = self.prior_L(L)
            
            # === Step 3: 更新 V ===
            V = self.step_V(k, L, Z_V)
            
            # 【核心修改 B】：主动归一化 V
            V = self.norm_V(V)
            
            # 此时 V 的数值会在 -3 到 3 之间波动，偶尔到 10
            # 绝对不会再撞到 500 的天花板
            V = torch.clamp(V, min=-50.0, max=50.0)
            
            Z_V = self.prior_V(V)

            # === Loss 计算 ===
            # 注意：因为 L 和 V 被 Norm 了，它们的数值范围变小了
            # 计算 Loss 时，Z_L 和 Z_V 也会适应这个范围，这是正确的
            F_1 = self.proj_96to1(F_val)
            loss_F_step = F.mse_loss(F_1, Z_F)
            loss_L_step = F.mse_loss(L, Z_L)
            loss_V_step = F.mse_loss(V, Z_V)
            
            loss_aux += loss_F_step +  loss_L_step + loss_V_step

        return F_val, V, loss_aux, 0.0
    

class DUNF(nn.Module):
    def __init__(self):
    # def __init__(self):
        super(DUNF, self).__init__()

        self.init = initModule()
        self.dum = DUM()
        self.PostProcess = PostModule()

    def forward(self,x):
        SO,DO_0 = self.init(x) #96,96,16
        SO = F.layer_norm(SO, SO.shape[1:])
        DO_0 = F.layer_norm(DO_0, DO_0.shape[1:])
        # print("DO_0 input:", SO.max(), DO_0.max())
        if torch.isnan(SO).any() or torch.isinf(SO).any() or torch.isnan(DO_0).any() or torch.isinf(DO_0).any():
            print(f"⚠️ Warning: SO,DO_0 contains NaN/Inf! Skipping batch.")
            
              # 直接跳过这个 batch，不送入网络
        F1,V,Loss,_ = self.dum(SO,DO_0)#96,96
        # print("F input:", F1.max(), V.max())
        if torch.isnan(F1).any() or torch.isinf(F1).any() or torch.isnan(V).any() or torch.isinf(V).any():
            print(f"⚠️ Warning: F,V,Loss contains NaN/Inf! Skipping batch.")
            
              # 直接跳过这个 batch，不送入网络
        x = self.PostProcess(F1,SO,V)
        if torch.isnan(x).any() or torch.isinf(x).any() :
            print(f"⚠️ Warning: x contains NaN/Inf! Skipping batch.")
            print("x input:", x.max())
        return x,Loss,0.0


if __name__ == '__main__':
    import torch
    from thop import profile
    import numpy as np

    # 1. 初始化输入和模型（与原代码一致）
    input_tensor = np.ones((1, 3, 256, 256), dtype=np.float32)
    input_tensor = torch.tensor(input_tensor, dtype=torch.float32, device='cpu')
    
    model = DUNF()
    model.eval()  # 关闭BatchNorm/Dropout，确保统计准确

    a,b,c = model(input_tensor)

    # 2. 统计整体FLOPs和参数量（兼容旧版thop，无多余参数）
    total_flops, total_params = profile(model, inputs=(input_tensor,))

    # 3. 纯Python提取模块FLOPs（不依赖pandas，避免KeyError）
    module_stats = []
    for name, module in model.named_modules():
        if hasattr(module, '__flops__') and module.__flops__ > 0:
            module_flop = module.__flops__
            module_param = sum(p.numel() for p in module.parameters())
            module_stats.append({
                'name': name,
                'flop_g': module_flop / 1e9,
                'ratio': (module_flop / total_flops) * 100,
                'param_m': module_param / 1e6
            })

    # 4. 打印整体统计（核心信息优先）
    print("="*80)
    print(f"【整体统计】")
    print(f"总FLOPs：{total_flops / 1e9:.2f} GFLOPS")
    print(f"总参数量：{total_params / 1e6:.2f} M")
    print("="*80)

    # 5. 打印子模块统计（有则输出，无则提示，不报错）
    if module_stats:
        # 按FLOPs降序排序，取前15个高占比模块
        module_stats_sorted = sorted(module_stats, key=lambda x: x['flop_g'], reverse=True)[:15]
        print("【Top 15 高FLOPs子模块】")
        print(f"{'模块名称':<40} {'FLOPs(GF)':<12} {'占比(%)':<10} {'参数量(M)':<10}")
        print("-"*80)
        for stats in module_stats_sorted:
            print(f"{stats['name']:<40} {stats['flop_g']:.3f} {'':<4} {stats['ratio']:.1f} {'':<5} {stats['param_m']:.3f}")
    else:
        print("【子模块统计提示】旧版thop未支持子模块FLOPs自动提取，仅展示顶层核心模块统计")
    print("="*80)

    # 6. 手动统计顶层核心模块（关键！确保核心模块FLOPs必出）
    print("【顶层核心模块统计】（手动精准统计，不受thop版本影响）")
    print("-"*80)

    # 定义辅助函数：单独统计某个模块的FLOPs（需传入该模块的真实输入）
    def get_module_flops(module, *inputs):
        """
        单独统计单个模块的FLOPs和参数量
        inputs：该模块forward需要的输入参数（按顺序传入）
        """
        module.eval()
        with torch.no_grad():
            # 用thop单独统计该模块
            flops, _ = profile(module, inputs=inputs)
            params = sum(p.numel() for p in module.parameters())
        return flops, params

    # （1）统计init模块（输入：原始input_tensor）
    init_flops, init_params = get_module_flops(model.init, input_tensor)
    print(f"init模块：")
    print(f"  FLOPs：{init_flops / 1e9:.2f} GFLOPS（占比：{init_flops/total_flops*100:.1f}%）")
    print(f"  参数量：{init_params / 1e6:.2f} M")
    print("-"*40)

    # （2）统计dum模块（输入：init模块的输出SO、DO_0）
    with torch.no_grad():
        SO, DO_0 = model.init(input_tensor)  # 获取init的真实输出，作为dum的输入
    dum_flops, dum_params = get_module_flops(model.dum, SO, DO_0)
    print(f"dum模块：")
    print(f"  FLOPs：{dum_flops / 1e9:.2f} GFLOPS（占比：{dum_flops/total_flops*100:.1f}%）")
    print(f"  参数量：{dum_params / 1e6:.2f} M")
    print("-"*40)

    # （3）统计PostProcess模块（输入：F、SO、V，需先获取dum的输出）
    with torch.no_grad():
        F,V,Loss,_= model.dum(SO, DO_0)  # 获取dum的真实输出，作为PostProcess的输入
    post_flops, post_params = get_module_flops(model.PostProcess, F,SO,V)
    print(f"PostProcess模块：")
    print(f"  FLOPs：{post_flops / 1e9:.2f} GFLOPS（占比：{post_flops/total_flops*100:.1f}%）")
    print(f"  参数量：{post_params / 1e6:.2f} M")
    print("="*80)