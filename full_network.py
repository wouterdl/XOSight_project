import torch 
import torch.nn as nn
from mti_net import *
from hrnet_backbone import *




class FullNet(nn.Module):
    '''
    Class that combines the backbone and MTI-net components, and adds the YOLO head to create the full network.

    INPUT:
    - a dict 'x' that contains the input data (x['rgb']) and the task labels (x['TASK_NAME']) for one sample

    OUTPUT:
    - a dict 'outputs' that contains all the predictions for the different tasks
    '''
    def __init__(self, cfg, device):

        super(FullNet, self).__init__()

        self.cfg = cfg
        self.input = input
        self.yolo = False
        self.att_channel0 = [36, 18, 18]
        self.att_channel1 = [72, 36, 36]
        self.att_channel2 = [144, 72, 72]
        self.att_channel3 = [288, 144, 144]

        self.pred_channel0 = 18
        self.pred_channel1 = 36
        self.pred_channel2 = 72
        self.pred_channel3 = 144

        self.backbone = hrnet_backbone.HighResolutionNet(cfg=self.cfg.HRnet_cfg).to(device)
        self.MTINet = MTINet(cfg=self.cfg.MTI_cfg, backbone=self.backbone).to(device)

        if 'bbox' in cfg.general_cfg.TASKS.NAMES:
            print('APPLYING YOLO HEAD')
            self.yolo = True

        if self.yolo == True:
            self.att0 = AttentionModule(channel=self.att_channel0, cfg=cfg).to(device)
            self.att1 = AttentionModule(channel=self.att_channel1, cfg=cfg).to(device)
            self.att2 = AttentionModule(channel=self.att_channel2, cfg=cfg).to(device)
            self.att3 = AttentionModule(channel=self.att_channel3, cfg=cfg).to(device)

            self.pred0 = ScalePrediction(in_channels=self.pred_channel0, cfg = cfg.yolo_cfg).to(device)
            self.pred1 = ScalePrediction(in_channels=self.pred_channel1, cfg = cfg.yolo_cfg).to(device)
            self.pred2 = ScalePrediction(in_channels=self.pred_channel2, cfg = cfg.yolo_cfg).to(device)
            self.pred3 = ScalePrediction(in_channels=self.pred_channel3, cfg = cfg.yolo_cfg).to(device)

        


    def forward(self, x):
        
        #Get the features from the backbone
        init_features = self.backbone(x['rgb'])

        # for tag, parm in self.backbone.named_parameters():
        #     if parm.grad == None:
        #         print('NONETYPE PARM: {}'.format(tag))
        #         print("requires_grad: ", parm.requires_grad)

        #print('init feature shape scale 0: {}'.format(init_features[1].shape))

        #Get output from the MTInet component
        mti_output = self.MTINet(init_features)
        outputs = mti_output.copy()

        if self.yolo == True:

            #Apply attention to the backbone and MTInet feature distillation
            scale_0_att = self.att0(init_features[0], mti_output['multi_scale_features']['scale_0']['bbox'])
            scale_1_att = self.att1(init_features[1], mti_output['multi_scale_features']['scale_1']['bbox'])
            scale_2_att = self.att2(init_features[2], mti_output['multi_scale_features']['scale_2']['bbox'])
            scale_3_att = self.att3(init_features[3], mti_output['multi_scale_features']['scale_3']['bbox'])

        

            #Make the YOLO predictions
            scale_0_pred = self.pred0(scale_0_att)
            scale_1_pred = self.pred1(scale_1_att)
            scale_2_pred = self.pred2(scale_2_att)
            scale_3_pred = self.pred3(scale_3_att)

            #Store the YOLO output
            yolo_output = {'bbox' :[scale_0_pred, scale_1_pred, scale_2_pred, scale_3_pred]}

            #combine predictions
            
            outputs.update(yolo_output)
        

        return outputs







        