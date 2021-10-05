import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os, os.path
import imageio
from hrnet_backbone import *
import hrnet_backbone
from mti_net import *
import mti_net
from easydict import EasyDict as edict
from network_modules import *
import network_modules
from train import *
from utils import *
from config import *
from full_network import *
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")


class WarehouseDataset(Dataset):
    def __init__(self, cfg):
        
        self.data_dtype = torch.float32
        #self.data_path = os.path.join(os.getcwd(), 'testing/output/Viewport')
        self.data_path = cfg['data_path']
        
        self.data_len = len([name for name in os.listdir(os.path.join(self.data_path ,'rgb')) if os.path.isfile(os.path.join(os.path.join(self.data_path, 'rgb'), name))])

        #changing np.load so allow_pickle=True
        np_load_old = np.load
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)    

        self.rgb_data = []
        self.bbox_2d_tight_data = []
        self.depth_data = []
        self.semantic_data = []
        self.anchors = cfg['anchors']
        self.anchors = torch.tensor(self.anchors[0] + self.anchors[1] + self.anchors[2] + self.anchors[3])
        self.num_anchors = self.anchors.shape[0]
        self.S = cfg['S']
        self.num_anchors_per_scale = self.num_anchors // 4
        print(self.num_anchors_per_scale)
        self.ignore_iou_thresh = cfg['iou_thresh']


        for i in range(self.data_len):
            print('reading data of sample {}'.format(i))
        
            self.rgb_data.append(imageio.imread(os.path.join(self.data_path, 'rgb/{}.png'.format(i))))
           
            
            self.bbox_2d_tight_data.append(np.load(os.path.join(self.data_path, 'bbox_2d_tight/{}.npy'.format(i))))
            #bbox data format: [?, prim path, str(label), ?, ?. int(label), x1, y1, x2, y2]
            self.depth_data.append(np.load(os.path.join(self.data_path, 'depth/{}.npy'.format(i))))
            self.semantic_data.append(np.load(os.path.join(self.data_path, 'semantic/{}.npy'.format(i))).astype('int32'))

            

        self.data_res = self.rgb_data[-1].shape[0]

        np.load = np_load_old

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):


        #Making BBox data suitable for YOLO head
        bbox_data = []

        #rearranging the bbox data to [class_label, x, y, width, height]
        for i in range(self.bbox_2d_tight_data[idx].shape[0]):

            class_label = self.bbox_2d_tight_data[idx][i][5]
            x_center = (self.bbox_2d_tight_data[idx][i][6] + self.bbox_2d_tight_data[idx][i][8])/2
            y_center = (self.bbox_2d_tight_data[idx][i][7] + self.bbox_2d_tight_data[idx][i][9])/2
            width = (self.bbox_2d_tight_data[idx][i][8] - self.bbox_2d_tight_data[idx][i][6])
            height = (self.bbox_2d_tight_data[idx][i][9] - self.bbox_2d_tight_data[idx][i][7])
            bbox_data.append(np.array([class_label, x_center/self.data_res, y_center/self.data_res, width/self.data_res, height/self.data_res]))

        targets = [torch.zeros((self.num_anchors // 4, S, S, cfg.general_cfg.no_classes+5)) for S in self.S]

        #Applying anchor boxes
        for box in bbox_data:
            iou_anchors = iou_width_height(torch.tensor(box[3:]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            
            class_label, x, y, width, height = box
            has_anchor = [False] * 4  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                #print(scale_idx)
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)  # which cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction




        item = {"rgb": torch.from_numpy(self.rgb_data[idx][:, :, 0:3]).type(self.data_dtype).permute(2, 0, 1), 
                "bbox": tuple(targets)[::-1],
                "depth": torch.from_numpy(self.depth_data[idx]).type(self.data_dtype), 
                "semantic": torch.from_numpy(self.semantic_data[idx]).type(torch.int64)}


        return item








if __name__ == '__main__':
    
    dataset = WarehouseDataset(cfg['dataset_cfg'])

    model = FullNet(cfg=cfg, device=device).to(device)

    print('bbox label shape: {}'.format(dataset[0]['bbox'][0].shape))


    temp_anchors = [
        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
        ]

    temp_S = [8, 16, 32]
    scaled_anchors = (
        torch.tensor(cfg.dataset_cfg.anchors)
        * torch.tensor(cfg.dataset_cfg.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(device)


    
    trainer = NetworkTrainer(model=model, dataset=dataset, tasks=cfg.general_cfg.TASKS.NAMES, loss_weights=cfg.general_cfg.loss_weights, max_epochs=250, scaled_anchors=scaled_anchors)
    
    
    

    trainer.train()


    writer = SummaryWriter()
    
    trainer.visualize(0, dataset, cfg.general_cfg.TASKS.NAMES, model=model)


    # testinput = dataset[0]
    # testinput['rgb'] = testinput['rgb'].unsqueeze(0)


    # for k, v in testinput.items():
    #                 if torch.is_tensor(testinput[k]):
    #                     testinput[k] = testinput[k].to(device)

    #                 elif isinstance(testinput[k], list):
    #                     #print(data[k])
    #                     for i in range(len(testinput[k])):
    #                         testinput[k][i] = testinput[k][i].to(device)
    # testoutput = model(testinput)
    # print(testoutput['bbox'][0].shape)
    



    

