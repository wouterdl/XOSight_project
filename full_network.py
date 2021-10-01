import torch 
import torch.nn as nn
from mti_net import *
from hrnet_backbone import *




class FullNet(nn.Module):
    def __init__(self, cfg, device):

        super(FullNet, self).__init__()

        self.cfg = cfg
        self.input = input
        self.att_channel0 = [18, 18, 18]
        self.att_channel1 = [36, 36, 36]
        self.att_channel2 = [72, 72, 72]
        self.att_channel3 = [144, 144, 144]

        self.pred_channel0 = 18
        self.pred_channel1 = 36
        self.pred_channel2 = 72
        self.pred_channel3 = 144

        self.backbone = hrnet_backbone.HighResolutionNet(cfg=self.cfg.HRnet_cfg).to(device)
        self.MTINet = MTINet(cfg=self.cfg.MTI_cfg, backbone=self.backbone).to(device)
        self.att0 = AttentionModule(channel=self.att_channel0, cfg=cfg).to(device)
        self.att1 = AttentionModule(channel=self.att_channel1, cfg=cfg).to(device)
        self.att2 = AttentionModule(channel=self.att_channel2, cfg=cfg).to(device)
        self.att3 = AttentionModule(channel=self.att_channel3, cfg=cfg).to(device)

        self.pred0 = ScalePrediction(in_channels=self.pred_channel0, cfg = cfg.yolo_cfg).to(device)
        self.pred1 = ScalePrediction(in_channels=self.pred_channel1, cfg = cfg.yolo_cfg).to(device)
        self.pred2 = ScalePrediction(in_channels=self.pred_channel2, cfg = cfg.yolo_cfg).to(device)
        self.pred3 = ScalePrediction(in_channels=self.pred_channel3, cfg = cfg.yolo_cfg).to(device)

        


    def forward(self, x):

        init_features = self.backbone(x['rgb'])

        #print('init feature shape scale 0: {}'.format(init_features[1].shape))

        #Get predictions of per pixel tasks
        mti_output = self.MTINet(init_features)

        #print('MTI features shape scale 0: {}'.format(mti_output[0].shape))

        #Get YOLO (bbox) predictions
        # scale_0_att = self.att(init_features[0], mti_output['deep_supervision']['scale_0'])
        # scale_1_att = self.att(init_features[1], mti_output['deep_supervision']['scale_1'])
        # scale_2_att = self.att(init_features[2], mti_output['deep_supervision']['scale_2'])
        # scale_3_att = self.att(init_features[3], mti_output['deep_supervision']['scale_3'])
        scale_0_att = self.att0(init_features[0])
        scale_1_att = self.att1(init_features[1])
        scale_2_att = self.att2(init_features[2])
        scale_3_att = self.att3(init_features[3])

        #print('attention features shape scale 0: {}'.format(scale_1_att.shape))

        scale_0_pred = self.pred0(scale_0_att)
        scale_1_pred = self.pred1(scale_1_att)
        scale_2_pred = self.pred2(scale_2_att)
        scale_3_pred = self.pred3(scale_3_att)

        #print('scaleprediction features shape scale 0: {}'.format(scale_1_pred.shape))

        yolo_output = {'bbox' :[scale_0_pred, scale_1_pred, scale_2_pred, scale_3_pred]}

        #combine predictions
        outputs = mti_output.copy()
        outputs.update(yolo_output)

        return outputs







        