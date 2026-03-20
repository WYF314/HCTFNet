import torch
import torch.nn as nn
import torch.nn.functional as F

from models.block.Base import Conv3Relu
from models.block.Drop import DropBlock
from models.block.Field import PPM, ASPP, SPP
from models.neck.SCN import AlignedModule,AlignedModulev2PoolingAtten

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3_bn_relu(in_planes, out_planes, stride=1, normal_layer=nn.BatchNorm2d):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            normal_layer(out_planes),
            nn.ReLU(inplace=True),
    )

class FPNNeck(nn.Module):
    def __init__(self, inplanes, neck_name='fpn+ppm+fuse'):
        super().__init__()
        self.stage1_Conv1 = Conv3Relu(inplanes * 2, inplanes)  # channel: 2*inplanes ---> inplanes
        self.stage2_Conv1 = Conv3Relu(inplanes * 4, inplanes * 2)  # channel: 4*inplanes ---> 2*inplanes
        self.stage3_Conv1 = Conv3Relu(inplanes * 8, inplanes * 4)  # channel: 8*inplanes ---> 4*inplanes
        self.stage4_Conv1 = Conv3Relu(inplanes * 16, inplanes * 8)  # channel: 16*inplanes ---> 8*inplanes

        
        self.scn1= AlignedModulev2PoolingAtten(inplanes * 1, inplanes * 1, inplanes )
        self.scn2= AlignedModulev2PoolingAtten(inplanes * 2, inplanes * 2, inplanes )
        self.scn3= AlignedModulev2PoolingAtten(inplanes * 4, inplanes * 4, inplanes )
        
       
        self.down3=conv3x3_bn_relu( inplanes * 8,inplanes * 4)
        self.down2=conv3x3_bn_relu( inplanes * 4,inplanes * 2)
        self.down1=conv3x3_bn_relu( inplanes * 2,inplanes * 1)
        
        self.scn41= AlignedModulev2PoolingAtten(inplanes , inplanes, inplanes)
        self.scn31= AlignedModulev2PoolingAtten(inplanes , inplanes, inplanes)
        self.scn21= AlignedModulev2PoolingAtten(inplanes , inplanes, inplanes)
        self.final_Conv5 = Conv3Relu(inplanes , inplanes)
        

        # PPM/ASPP比SPP好
        if "+ppm+" in neck_name:
            self.expand_field = PPM(inplanes * 8)
        elif "+aspp+" in neck_name:
            self.expand_field = ASPP(inplanes * 8)
        elif "+spp+" in neck_name:
            self.expand_field = SPP(inplanes * 8)
        else:
            self.expand_field = None

        if "fuse" in neck_name:
            self.stage2_Conv3 = Conv3Relu(inplanes * 2, inplanes)   # 降维
            self.stage3_Conv3 = Conv3Relu(inplanes * 4, inplanes)
            self.stage4_Conv3 = Conv3Relu(inplanes * 8, inplanes)

            self.final_Conv = Conv3Relu(inplanes * 4, inplanes)
            

            self.fuse = True
        else:
            self.fuse = False

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        if "drop" in neck_name:
            rate, size, step = (0.15, 7, 30)
            self.drop = DropBlock(rate=rate, size=size, step=step)
        else:
            self.drop = DropBlock(rate=0, size=0, step=0)

    def forward(self, ms_feats):
        fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4 = ms_feats
        change1_h, change1_w = fa1.size(2), fa1.size(3)

        [fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4] = self.drop([fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4])  # dropblock

        change1 = self.stage1_Conv1(torch.cat([fa1, fb1], 1))  # inplanes
        change2 = self.stage2_Conv1(torch.cat([fa2, fb2], 1))  # inplanes * 2
        change3 = self.stage3_Conv1(torch.cat([fa3, fb3], 1))  # inplanes * 4
        change4 = self.stage4_Conv1(torch.cat([fa4, fb4], 1))  # inplanes * 8

        if self.expand_field is not None:
            change4 = self.expand_field(change4)
       
        change3=self.scn3(change3, self.down3(change4))
        change2=self.scn2(change2, self.down2(change3))
        change1=self.scn1(change1, self.down1(change2))

        if self.fuse:
           
            change4=self.scn41(change1, self.stage4_Conv3(change4))
            change3=self.scn31(change1, self.stage3_Conv3(change3))
            change2=self.scn21(change1, self.stage2_Conv3(change2))

            [change1, change2, change3, change4] = self.drop([change1, change2, change3, change4])  # dropblock

            change = self.final_Conv(torch.cat([change1, change2, change3, change4], 1))
           
        else:
            change = change1
            change=self.final_Conv5(change1)

        return change

