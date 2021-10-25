import torch
import torch.nn as nn
from hrnet_backbone import *
import hrnet_backbone
import torch.nn.functional as F


class BasicBlock(nn.Module):
    '''
    Basic block used in MTI-net
    '''
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = hrnet_backbone.conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = hrnet_backbone.conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SEBlock(nn.Module):
    """ Squeeze-and-excitation block """
    def __init__(self, channels, r=16):
        super(SEBlock, self).__init__()
        self.r = r 
        self.squeeze = nn.Sequential(nn.Linear(channels, channels//self.r),
                                     nn.ReLU(),
                                     nn.Linear(channels//self.r, channels),
                                     nn.Sigmoid())

    def forward(self, x):
        B, C, H, W = x.size()
        squeeze = self.squeeze(torch.mean(x, dim=(2,3))).view(B,C,1,1)
        return torch.mul(x, squeeze)

class SABlock(nn.Module):
    """ Spatial self-attention block """
    def __init__(self, in_channels, out_channels):
        super(SABlock, self).__init__()
        self.attention = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                        nn.Sigmoid())
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

    def forward(self, x):
        attention_mask = self.attention(x)
        features = self.conv(x)
        return torch.mul(features, attention_mask)

class MultiTaskDistillationModule(nn.Module):
    """
        Perform Multi-Task Distillation
        We apply an attention mask to features from other tasks and
        add the result as a residual.
    """
    def __init__(self, tasks, auxilary_tasks, channels):
        super(MultiTaskDistillationModule, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.self_attention = {}
        
        for t in self.tasks:
            other_tasks = [a for a in self.auxilary_tasks if a != t]
            self.self_attention[t] = nn.ModuleDict({a: SABlock(channels, channels) for a in other_tasks})
        self.self_attention = nn.ModuleDict(self.self_attention)


    def forward(self, x):
        #print(self.self_attention[t][a](x['features_%s' %(a)]) for a in self.auxilary_tasks if a!= t)
        adapters = {t: {a: self.self_attention[t][a](x['features_%s' %(a)]) for a in self.auxilary_tasks if a!= t} for t in self.tasks}
        out = {}
        for t in self.tasks:
            if len(self.auxilary_tasks) == 1 and not t == 'bbox':
                out[t] = x['features_%s' %(t)]
            elif not t == 'bbox':
                out[t] = x['features_%s' %(t)] + torch.sum(torch.stack([v for v in adapters[t].values()]), dim=0)
            else:
                
                out[t] = torch.sum(torch.stack([v for v in adapters[t].values()]), dim=0)
        
        return out

class HighResolutionHead(nn.Module):

    def __init__(self, backbone_channels, num_outputs):
        super(HighResolutionHead, self).__init__()
        last_inp_channels = sum(backbone_channels)
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum = 0.1),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels= num_outputs,
                kernel_size= 1,
                stride = 1,
                padding = 0))
    
    def forward(self, x):
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], (x0_h, x0_w), mode='bilinear')
        x2 = F.interpolate(x[2], (x0_h, x0_w), mode='bilinear')
        x3 = F.interpolate(x[3], (x0_h, x0_w), mode='bilinear')

        x = torch.cat([x[0], x1, x2, x3], 1)
        x = self.last_layer(x)
        return x 

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)

class ScalePrediction(nn.Module):
    '''
    YOLO prediction module at a specified scale.
    '''
    def __init__(self, in_channels, cfg):
        super(ScalePrediction, self).__init__()
        self.num_classes = cfg.num_classes
        self.anchors_per_scale = cfg.anchors_per_scale

        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2*in_channels, kernel_size=3, padding=1),
            CNNBlock(2*in_channels, (self.num_classes + 5) * 3, bn_act=False, kernel_size=1),
        )
        
    def forward(self, x):
        return (
            self.pred(x)
                .reshape(x.shape[0], self.anchors_per_scale, self.num_classes + 5, x.shape[2], x.shape[3])
                .permute(0, 1, 3, 4, 2)
        )


class AttentionModule(nn.Module):
    '''
    Attention module used to link the YOLO head to the rest of the network.
    INPUT:
    - input1: a [N, C, H, W] tensor, coming from the bacbone
    - input2: a [N, C, H, W] tensor, coming from the MTInet feature distillation

    OUTPUT:
    - out: a [N, C, H, W] tensor: the inputs combined with attention applied
    '''
    def __init__(self, channel, cfg):
        super(AttentionModule, self).__init__()

        self.channel = channel

        self.conv1 = nn.Conv2d(in_channels=self.channel[0], out_channels=self.channel[1], kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=self.channel[1], out_channels=self.channel[2], kernel_size=1, padding=0)
        self.BN1 = nn.BatchNorm2d(channel[1])
        self.BN2 = nn.BatchNorm2d(channel[2])
        self.ReLU = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input1, input2=None):
        #print(input1.shape)
        #print(input2.keys())
        #merge = torch.cat((input1, input2), dim=0)

        if not input2 == None: #If 2 inputs are given
            merge = torch.cat((input1, input2), dim=1)
        else:   #If only 1 input is given
            merge = input1
        #print(merge.shape)
        xt = self.conv1(merge)
        x = self.BN1(xt)
        x = self.ReLU(x)
        x = self.conv2(x)
        x = self.BN2(x)
        x = self.sigmoid(x)

        out = xt * x
        #out = x
        #print(out.shape)
        return out



