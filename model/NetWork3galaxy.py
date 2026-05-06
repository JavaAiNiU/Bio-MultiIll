from turtle import forward
import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
# from einops import rearrange
from timm.layers import DropPath, to_2tuple, trunc_normal_
import math



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
        
        padding = kernel_size // 2
        
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
    

class MultiKernelDiff(nn.Module):
    def __init__(self, in_channels):
        super(MultiKernelDiff, self).__init__()
        
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
        
        self.Lcam = CAM(channel=in_channels*3)
        self.Scam = CAM(channel=in_channels*3)

    def forward(self, x, y):

        
        Soutx3 = self.conv3_1(x)
        Souty3 = self.conv3_2(y)

        Soutx5 = self.conv5_1(x)
        Souty5 = self.conv5_2(y)

        Soutx7 = self.conv7_1(x)
        Souty7 = self.conv7_2(y)

        Sdiff1 = Soutx3 - Souty3 
        Sdiff2 = Soutx5 - Souty5 
        Sdiff3 = Soutx7 - Souty7  

        Sresult = torch.cat([Sdiff1, Sdiff2, Sdiff3], dim=1)
        Sresult = self.Scam(Sresult)

        
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
        Lresult = self.Lcam(Lresult)
        
        SO = torch.cat([Sresult, Lresult], dim=1)  
        DO = Sresult - Lresult   
        
        return SO,DO


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
        
        pool1 = F.max_pool2d(L1,kernel_size=2)  
        
        
        L2 = self.conv2(pool1)  
        
        pool2 = F.max_pool2d(L2,kernel_size=2)  
        
        
        L3 = self.conv3(pool2)  
        
        L4 = F.max_pool2d(L3,kernel_size=2)  
        
        
        # L4 = self.conv4(pool3)  
        
        
        # poolL = F.max_pool2d(L4,kernel_size=2)  
        
        return L1,L2,L3,L4


class UNet(nn.Module):  
    def __init__(self, in_channels=96):  
        super(UNet, self).__init__()
        self.in_channels = in_channels  
        
        self.low_Encoder = Low_Encoder(in_channels)  
        
        
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),  
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),  
            LeakyReLU(),
        )
        self.conv_up1 = UNetConvBlock(512, 256)  
        
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            LeakyReLU(),
        )
        self.conv_up2 = UNetConvBlock(256, 128)  
        
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            LeakyReLU(),
        )
        self.conv_up3 = UNetConvBlock(128, 128)  
        

        
        
        self.convlast1 = self._build_convlast1(
            start_channel=128,  
            target_channel=in_channels  
        )

    def _build_convlast1(self, start_channel, target_channel):
        """
        私有方法：动态构建convlast1模块
        结构规则：每步保持「3x3卷积（保通道）→LeakyReLU→1x1卷积（降维）→LeakyReLU」
        降维逻辑：从start_channel开始，每次优先除以2，最后一步调整至target_channel
        """
        layers = []
        current_ch = start_channel  

        
        while current_ch > target_channel:
            
            next_ch = current_ch // 2 if (current_ch // 2) >= target_channel else target_channel
            
            
            layers.extend([
                
                nn.Conv2d(current_ch, current_ch, kernel_size=3, stride=1, padding=1),
                LeakyReLU(),  
                
                nn.Conv2d(current_ch, next_ch, kernel_size=1, stride=1, padding=0),
                LeakyReLU()  
            ])
            
            current_ch = next_ch  

        
        return nn.Sequential(*layers)

    def forward(self, x):
        """前向传播：编码器→三次上采样拼接→动态降维→输出"""
        
        L1, L2, L3, L4 = self.low_Encoder(x)

        
        x = self.up1(L4)  
        x = torch.cat([x, L3], dim=1)  
        x = self.conv_up1(x)  

        
        x = self.up2(x)  
        x = torch.cat([x, L2], dim=1)  
        x = self.conv_up2(x)  

        
        x = self.up3(x)  
        x = torch.cat([x, L1], dim=1)  
        x = self.conv_up3(x)  

        
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
            
            nn.Conv2d(
                in_channels=in_chans, 
                out_channels=mid_dim, 
                kernel_size=patch_size, 
                stride=stride,
                padding=(patch_size[0] // 2, patch_size[1] // 2)
            ),
            LeakyReLU(),  
            
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
    
    def forward(self, x1, x2, H, W):
        B, N, D = x1.size()  # (Batch, Tokens, Embedding dim)
        # Re-arrange into a (Batch, Embedding dim, Tokens)
        
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


class PsotConvBlock(nn.Module):
    def __init__(self, in_channel,out_channel):  
        
        super(PsotConvBlock, self).__init__()
        
        
        
        mid_channel = max(in_channel // 2, 32)
        
        
        self.conv_block = torch.nn.Sequential(
            
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1),
            LeakyReLU(),
            nn.Conv2d(in_channels=in_channel, out_channels=mid_channel, kernel_size=1, stride=1, padding=0),
            LeakyReLU(),  

            nn.Conv2d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, padding=1),
            LeakyReLU(),  
            nn.Conv2d(in_channels=mid_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            LeakyReLU(),  

            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
            LeakyReLU(),  
        )
    
    def forward(self, x):
        
        return self.conv_block(x)
    

class PsotConvBlock2(nn.Module):
    def __init__(self, in_channel,out_channel):  
        
        super(PsotConvBlock2, self).__init__()

        
        self.conv_block = torch.nn.Sequential(
            
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

        SO = self.cam(SO)
        SO = self.SO(F*SO)
        V =self.DO(V)# 16
        V_e = self.patch_embed_C(V)#,64
        SO_e = self.patch_embed_C(SO)#,64


        SO_e = self.fuse(V_e,SO_e)

        
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
            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0),  
            LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0),
            LeakyReLU()
        )

        self.MSO_RG = MultiKernelDiff(16)
        self.MSO_YB = MultiKernelDiff(16)

        self.Convg2 = nn.Sequential(
            nn.Conv2d(192, 128 , kernel_size=1, stride=1, padding=0),
            LeakyReLU(),
            nn.Conv2d(128, 96, kernel_size=1, stride=1, padding=0),  
            LeakyReLU(),
            nn.Conv2d(96, 96, kernel_size=1, stride=1, padding=0),
            LeakyReLU()
        )

        
        
        

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
        r,G1,G2,b = self.split_feature_map(x) 
        g = torch.cat([G1,G2],dim=1)#
        g = self.Convg(g)#32->16

        SO_RG,DO_RG = self.MSO_RG(r,g) #96,48
        SO_YB,DO_YB = self.MSO_YB(b,(r+g)/2) #96,48

        

        return self.Convg2(torch.cat([SO_RG,SO_YB],dim=1)),torch.cat([DO_RG,DO_YB],dim=1)#,W #96,96,16
    

# class FkCalculator(nn.Module):
#     """

#     F_k = ( (L_{k-1} ⊙ S) + 2μ_F * σ_1^2 * Z_{F, k-1} ) / ( (S ⊙ S) + 2μ_F * σ_1^2 )
    
#     Args:



#     """
class FkCalculator(nn.Module):
    def __init__(self, sigma1_init: float = 1.0, mu_f_init: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.sigma1 = nn.Parameter(torch.tensor(sigma1_init, dtype=torch.float32), requires_grad=True)
        self.mu_f = nn.Parameter(torch.tensor(mu_f_init, dtype=torch.float32), requires_grad=True)
        self.eps = eps

    def forward(self, k, S, L_prev, Z_f_prev):
        if k <= 0: raise ValueError(f"k must be > 0")

        
        
        mu_f_safe = F.softplus(self.mu_f)
        sigma1_safe = self.sigma1 
        
        coeff = 2 * mu_f_safe * sigma1_safe.pow(2)

        
        
        S_safe = torch.clamp(S, min=-100.0, max=100.0) 

        numerator = (L_prev * S_safe) + (coeff * Z_f_prev)

        
        denominator = (S_safe * S_safe) + coeff + self.eps
        
        return numerator / denominator


class VkMultiConv(nn.Module):
    """
    保持输入输出通道数一致、特征维度一致的卷积模块
    结构: Conv(3x3) -> LeakyReLU -> Conv(3x3) -> LeakyReLU
    """
    def __init__(self, in_channels, kernel_size=3, negative_slope=0.2):
        super(VkMultiConv, self).__init__()
        
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

#     V_k = ( Net(L_k) + 2 * σ_2^2 * μ_V * Z_{V, k-1} ) / ( 1 + 2 * σ_2^2 * μ_V )
    

#     """
#     def __init__(self, channels: int, sigma2_init: float = 1.0, mu_v_init: float = 0.1, eps: float = 1e-8):
#         """
#         Args:


#         """
#         super().__init__()
        

#         self.KL_net = VkMultiConv(in_channels=channels, kernel_size=3)
        

#         self.sigma2 = nn.Parameter(
#             torch.tensor(sigma2_init, dtype=torch.float32),
#             requires_grad=True
#         )
        

#         self.mu_v = nn.Parameter(
#             torch.tensor(mu_v_init, dtype=torch.float32),
#             requires_grad=True
#         )
        
#         self.eps = eps

#     def forward(
#         self, 



#     ) -> torch.Tensor:
#         """


#         """

#         if k <= 0:

#         if Lk.shape != Zv_prev.shape:




#         conv_term = self.KL_net(Lk)
        

#         coeff = 2 * self.sigma2.pow(2) * self.mu_v
        

#         # Term 1: Net(L_k)
#         # Term 2: coeff * Z_{V, k-1}
#         numerator = conv_term + (coeff * Zv_prev)


#         # Term 1: 1
#         # Term 2: coeff
#         denominator = 1 + coeff + self.eps
        

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
        
        
        mu_v_safe = F.softplus(self.mu_v)
        
        coeff = 2 * self.sigma2.pow(2) * mu_v_safe
        
        numerator = conv_term + (coeff * Zv_prev)

        
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
            
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding=padding),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            
            
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

        
        feat_L = self.L_feature_net(Lk_prev)
        down_L = F.max_pool2d(feat_L, kernel_size=2)
        down_V = self.V_down_net(Vk_prev)
        residual_low = down_V - down_L
        residual_high = self.up_restore(residual_low)
        
        if residual_high.shape != S.shape:
             residual_high = F.interpolate(residual_high, size=S.shape[2:], mode='bilinear', align_corners=False)

        
        
        
        mu_l_safe = F.softplus(self.mu_l)
        coeff_mu = 2 * mu_l_safe * self.sigma1.pow(2)

        
        sigma2_sq = self.sigma2.pow(2) + 1e-4  
        
        sigma_ratio = self.sigma1.pow(2) / sigma2_sq
        
        
        
        
        
        sigma_ratio = torch.clamp(sigma_ratio, max=50.0) 

        numerator = (Fk * S) + (coeff_mu * ZL) + (sigma_ratio * residual_high)
        denominator = 1 + coeff_mu + self.eps

        return numerator / denominator

    
# class LkCalculator(nn.Module):
#     """

    




#     """
#     def __init__(self, channels: int, sigma1_init: float = 1.0, sigma2_init: float = 1.0, mu_l_init: float = 0.1, eps: float = 1e-8):
#         super().__init__()
        

#         self.L_feature_net = MultiConv(channels)
        

#         self.V_down_net = DownSampleConv(channels)
        

#         self.up_restore = nn.Sequential(
#             nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2),
#             nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True)
#         )
        

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


#     ) -> torch.Tensor:
        
#         if k <= 0: raise ValueError("k must be > 0")



#         # Output: [B, C, H/2, W/2]
#         feat_L = self.L_feature_net(Lk_prev)
#         down_L = F.max_pool2d(feat_L, kernel_size=2)
        


#         # Output: [B, C, H/2, W/2]
#         down_V = self.V_down_net(Vk_prev)


#         if down_V.shape != down_L.shape:





#         residual_low = down_V - down_L
        

#         residual_high = self.up_restore(residual_low)
        

#         if residual_high.shape != S.shape:

#             residual_high = F.interpolate(residual_high, size=S.shape[2:], mode='bilinear', align_corners=False)


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

        
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        
        # 2. ResNet Layers
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2) # Downsample /2
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2) # Downsample /4
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2) # Downsample /8

        
        
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
        
        
        out = self.up2(out)    # [B, 128, H/2, W/2]
        out = self.up1(out)    # [B, 64, H, W]
        

        residual = self.final_conv(out)
        
        return x + residual
    
class DUM(nn.Module):
    def __init__(self):
        super(DUM, self).__init__()
        
        
        self.step_F = FkCalculator()
        self.step_L = LkCalculator(channels=96)
        self.step_V = VkCalculator(channels=96)

        
        self.prior_F = Spatial_Attention() 
        self.prior_L = ResNet18_Prior(in_channels=96, out_channels=96)
        self.prior_V = UNet(in_channels=96)

        
        self.proj_96to1 = nn.Conv2d(96, 1, kernel_size=1, stride=1, padding=0, bias=True)
        
        
        self.w_F = nn.Parameter(torch.tensor(1.0))
        self.w_L = nn.Parameter(torch.tensor(1.0))
        self.w_V = nn.Parameter(torch.tensor(1.0))

        self.norm_L = nn.GroupNorm(num_groups=16, num_channels=96)
        self.norm_V = nn.GroupNorm(num_groups=16, num_channels=96)

    def forward(self, SO, DO_init):
        
        S = SO
        L = S.clone()
        V = DO_init.clone()
        F_val = torch.ones_like(S)
        
        Z_L = L.clone()
        Z_V = V.clone()
        Z_F = F_val.clone()
        
        loss_aux = 0.0

        
        for i in range(3):
            k = i + 1 
            
            
            F_val = self.step_F(k, S, L, Z_F)
            
            
            F_val = torch.clamp(F_val, min=0.01, max=10.0) 
            Z_F = self.prior_F(F_val)
            
            
            L = self.step_L(k, S, F_val, Z_L, L, V)
            
            
            
            L = self.norm_L(L) 
            
            
            L = torch.clamp(L, min=-50.0, max=50.0)
            
            Z_L = self.prior_L(L)
            
            
            V = self.step_V(k, L, Z_V)
            
            
            V = self.norm_V(V)
            
            
            
            V = torch.clamp(V, min=-50.0, max=50.0)
            
            Z_V = self.prior_V(V)


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
            
              
        F1,V,Loss,_ = self.dum(SO,DO_0)#96,96
        # print("F input:", F1.max(), V.max())
        if torch.isnan(F1).any() or torch.isinf(F1).any() or torch.isnan(V).any() or torch.isinf(V).any():
            print(f"⚠️ Warning: F,V,Loss contains NaN/Inf! Skipping batch.")
            
              
        x = self.PostProcess(F1,SO,V)
        if torch.isnan(x).any() or torch.isinf(x).any() :
            print(f"⚠️ Warning: x contains NaN/Inf! Skipping batch.")
            print("x input:", x.max())
        return x,Loss,0.0


if __name__ == '__main__':
    import torch
    from thop import profile
    import numpy as np

    
    input_tensor = np.ones((1, 3, 256, 256), dtype=np.float32)
    input_tensor = torch.tensor(input_tensor, dtype=torch.float32, device='cpu')
    
    model = DUNF()
    model.eval()  

    a,b,c = model(input_tensor)

    
    total_flops, total_params = profile(model, inputs=(input_tensor,))

    
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

    
    print("="*80)
    print(f"【整体统计】")
    print(f"总FLOPs：{total_flops / 1e9:.2f} GFLOPS")
    print(f"总参数量：{total_params / 1e6:.2f} M")
    print("="*80)
