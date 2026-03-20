import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.block.Base import Conv3Relu, Conv1Relu
from models.block.Drop import DropBlock
from models.block.ConvCABlock import ConvCABlock
from models.block.Attention import *
from models.block.FFTSA import SpatialFrequencyAttention
from models.block.DCT import DCT
    

class CrossSpatialAttention(nn.Module):
    def __init__(self, in_dim):
        super(CrossSpatialAttention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.query_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        m_batchsize, C, height, width = x1.size() #256 64 64    

        q1 = self.query_conv1(x1).view(m_batchsize, -1, width*height).permute(0, 2, 1)  #4096,32
        k1 = self.key_conv1(x1).view(m_batchsize, -1, width*height) #32,4096
        v1 = self.value_conv1(x1).view(m_batchsize, -1, width*height)   #256,4096

        q2 = self.query_conv2(x2).view(m_batchsize, -1, width*height).permute(0, 2, 1)  #4096,32
        k2 = self.key_conv2(x2).view(m_batchsize, -1, width*height) #32,4096
        v2 = self.value_conv2(x2).view(m_batchsize, -1, width*height)   #256,4096

        energy1 = torch.bmm(q1, k2) #4096 4096
        attention1 = self.softmax(energy1)
        out1 = torch.bmm(v2, attention1.permute(0, 2, 1)) #256 4096
        out1 = out1.view(m_batchsize, C, height, width) #256 64 64

        energy2 = torch.bmm(q2, k1)
        attention2 = self.softmax(energy2)
        out2 = torch.bmm(v1, attention2.permute(0, 2, 1))
        out2 = out2.view(m_batchsize, C, height, width)

        out1 = x1 + self.gamma1*out1  #gamma 1个
        out2 = x2 + self.gamma2*out2

        return out1, out2
    

class CrossFusionBlock(nn.Module):
    def __init__(self, ch): 
        super().__init__()
        self.csatt = CrossSpatialAttention(ch)
        self.fconv = nn.Conv2d(2*ch, ch, (1, 1), bias=False)
        self.num = 0

    def forward(self, fa, fb):
        fa, fb = self.csatt(fa, fb)
        x = torch.cat([fa, fb], 1)
        x = self.fconv(x)
        return x

class SFA(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.fftsa = SpatialFrequencyAttention(in_channel)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, (1,1), padding=0, 
                    groups=in_channel, bias=False),
            nn.Conv2d(in_channel, in_channel, (1,1), bias=False)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, (3,3), padding=1, 
                    groups=in_channel, bias=False),
            nn.Conv2d(in_channel, in_channel, (1,1), bias=False)
        )
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(num_features=in_channel)
        self.num = 0

    def forward(self, x): 
        xa, wa = self.fftsa(x)
        xa_conv = self.conv3(x)
        xa_conv = self.gelu(xa_conv)
        xa_conv = self.conv1(xa_conv)
        xa_conv = self.bn(xa_conv)
        xa_conv = self.gelu(xa_conv)
        x = x + xa * xa_conv
        return x


class RCA(nn.Module):
    """in/out channel should be same number"""
    def __init__(self, channel):
        super().__init__()

        self.convca = ConvCABlock(channel) 
        self.relu = nn.ReLU(inplace=False)
        self.bn = nn.BatchNorm2d(channel)
        self.conv = nn.Conv2d(channel, channel, (1, 1), bias=False)

    def forward(self, x):
        x = self.convca(x)
        return x

class DFE(nn.Module):
    """DCT Feature Enhance"""
    def __init__(self, channel):
        super().__init__()
        self.dfe = DCT()
        self.relu = nn.ReLU(inplace=False)
        self.bn = nn.BatchNorm2d(channel)
        self.conv = nn.Conv2d(channel, channel, (1, 1), bias=False)

    def forward(self, x):
        x_e = self.dfe(x)
        x1 = x * x_e
        x1 = self.conv(x1)
        x2 = self.bn(x_e)
        x2 = self.relu(x2)
        x2 = x1 + x2
        return x1, x2

class FPNNeck(nn.Module):
    def __init__(self, inplanes, neck_name='fpn+ppm+fuse'):
        super().__init__()

        self.fsa1 = SFA(64)
        self.fsa2 = SFA(128)
        self.fsa3 = SFA(256)
        self.fsa4 = SFA(512)
        self.rca1 = RCA(64)
        self.rca2 = RCA(128)
        self.rca3 = RCA(256)
        self.rca4 = RCA(512)
        self.dfe1 = DFE(64)
        self.dfe2 = DFE(128)
        self.dfe3 = DFE(256)
        self.dfe4 = DFE(512)

        self.cfm1 = CrossFusionBlock(64)
        self.cfm2 = CrossFusionBlock(128)
        self.cfm3 = CrossFusionBlock(256)
        self.cfm4 = CrossFusionBlock(512)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv1 = nn.Conv2d(64, 64, (1, 1), bias=False)
        self.conv2 = nn.Conv2d(128, 128, (1, 1), bias=False)
        self.conv3 = nn.Conv2d(256, 256, (1, 1), bias=False)
        self.conv4 = nn.Conv2d(512, 512, (1, 1), bias=False)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.Final_Conv = Conv3Relu(960, 64)

        rate, size, step = (0.3, 5, 25)
        self.drop = DropBlock(rate=rate, size=size, step=step)

    def forward(self, ms_feats):
        fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4 = ms_feats
        change1_c, change1_h, change1_w = fa1.size(1), fa1.size(2), fa1.size(3)

        [fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4] = self.drop([fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4])  # dropblock

        change1 = self.cfm1(fa1, fb1)
        change2 = self.cfm2(fa2, fb2)
        change3 = self.cfm3(fa3, fb3)
        change4 = self.cfm4(fa4, fb4)

        change1_l = self.fsa1(change1)
        change2_l = self.fsa2(change2)
        change3_l = self.fsa3(change3)
        change4_l = self.fsa4(change4)
        change1_g = self.rca1(change1)
        change2_g = self.rca2(change2)
        change3_g = self.rca3(change3)
        change4_g = self.rca4(change4)

        change1_g = self.bn1(change1_g)
        change2_g = self.bn2(change2_g)
        change3_g = self.bn3(change3_g)
        change4_g = self.bn4(change4_g)
        change1_g = F.gelu(change1_g)
        change2_g = F.gelu(change2_g)
        change3_g = F.gelu(change3_g)
        change4_g = F.gelu(change4_g)
        change1_l, change1_e  = self.dfe1(change1_l)
        change2_l, change2_e  = self.dfe2(change2_l)
        change3_l, change3_e  = self.dfe3(change3_l)
        change4_l, change4_e  = self.dfe4(change4_l)

        change1 = change1_g + change1_e + change1
        change2 = change2_g + change2_e + change2
        change3 = change3_g + change3_e + change3
        change4 = change4_g + change4_e + change4
        change1 = self.conv1(change1)
        change2 = self.conv2(change2)
        change3 = self.conv3(change3)
        change4 = self.conv4(change4)

        change1 = change1_l + change1
        change2 = change2_l + change2
        change3 = change3_l + change3
        change4 = change4_l + change4

        change2 = self.up(change2)
        change3 = self.up4(change3)
        change4 = self.up8(change4)
        
        change = self.Final_Conv(torch.cat([change1, change2, change3, change4], 1))

        return change
