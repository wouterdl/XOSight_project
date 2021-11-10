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
from skimage.color import rgb2gray
import cv2
import time

class WarehouseDataset(Dataset):
    '''
    Class that turns the generated isaac-sim dataset into a Pytorch dataset
    INPUT:
    for every task a folder 'TASK_NAME', containing .npy files with labels for every sample

    OUTPUT:
    a pytorch iterable dataset with for every index:
    - item: a dict, where item['rgb'] is the input data, and item['TASK'] the labels for task TASK:
        - 'rgb' : torch.tensor[C, H, W]
        - task 'semantic': torch.tensor[H, W]
        - task 'depth': torch.tensor[H, W]
        - task 'bbox' : tuple(torch.tensor[anchors_per_scale, S, S, no_classes + 5] for every scale)

    '''
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


        #Reading and storing the data from the npy files
        for i in range(self.data_len):
        #for i in range(10):
            print('reading data of sample {}'.format(i))
        
            self.rgb_data.append(imageio.imread(os.path.join(self.data_path, 'rgb/{}.png'.format(i))))
           
            boxes = np.load(os.path.join(self.data_path, 'bbox_2d_tight/{}.npy'.format(i)))
            #print(type(boxes))
            boxes_filtered = np.copy(boxes)
            lenn = len(boxes)
            del_indices = []
            print('lenn : {}'.format(lenn))
            for j in range(lenn):
                #print('index: {}'.format(j))
                #print('len boxes: {}'.format(len(boxes)))
                if boxes[j][5] > 2:
                    del_indices.append(j)
                    #boxes_filtered = np.delete(boxes_filtered, i, 0)
                    #lenn = len(boxes)
                    #boxes_filtered = np.append(boxes_filtered, boxes[i])

            boxes = np.delete(boxes, del_indices, 0)
            self.bbox_2d_tight_data.append(boxes)
            #bbox data format: [?, prim path, str(label), ?, ?. int(label), x1, y1, x2, y2]
            self.depth_data.append(np.load(os.path.join(self.data_path, 'depth/{}.npy'.format(i))))
            self.semantic_data.append(np.load(os.path.join(self.data_path, 'semantic/{}.npy'.format(i))).astype('int32'))

            

        self.data_res = self.rgb_data[-1].shape[0]

        np.load = np_load_old
        class_list = []

        for i in range(len(self.bbox_2d_tight_data)):
            for j in range(len(self.bbox_2d_tight_data[i])):
                class_list.append(self.bbox_2d_tight_data[i][j][5])



        class_list = list(set(class_list))


        print('UNIQUE CLASSES IN DATASET: {}'.format(class_list))
    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):


        ###Making BBox data suitable for YOLO head###
        bbox_data = []

        #rearranging the bbox data to [class_label, x, y, width, height]
        for i in range(self.bbox_2d_tight_data[idx].shape[0]):
            if not self.bbox_2d_tight_data[idx][i][5] == 1:
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



        ###creating the item dict that is returned by the function
        r, g, b = cv2.split(self.rgb_data[idx][:, :, 0:3])
        img = cv2.merge([r,r,r])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        item = {"rgb": torch.from_numpy(self.rgb_data[idx][:, :, 0:3]).type(self.data_dtype).permute(2, 0, 1), 
                #"rgb": torch.from_numpy(gray).unsqueeze(0).type(self.data_dtype), 
                "bbox": tuple(targets)[::-1],
                "depth": torch.from_numpy(self.depth_data[idx]).type(self.data_dtype), 
                "semantic": torch.from_numpy(self.semantic_data[idx]).type(torch.int64)}


        return item



if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = WarehouseDataset(cfg['dataset_cfg'])

    model = FullNet(cfg=cfg, device=device).to(device)

    max_time = 0.
    avg_time = 0.
    time_list = []

    

    for j in range(len(dataset)):
        
        groundtruth = dataset[j]
        for k, v in groundtruth.items():
            if torch.is_tensor(groundtruth[k]):
                groundtruth[k] = torch.unsqueeze(v, 0).to(device)

            elif isinstance(groundtruth[k], list):
                        #print(data[k])
                        for i in range(len(groundtruth[k])):
                            groundtruth[k][i] = groundtruth[k][i].to(device)
        start = time.time()
        output = model(groundtruth)

        exec_time = (time.time()-start)
        time_list.append(exec_time)
        if exec_time > max_time:
            max_time = exec_time

    avg_time = sum(time_list)/len(time_list)

    print('RESULTS:')
    print('AVG TIME: {} s MAX TIME: {} s'.format(avg_time, max_time))
    print('AVG FREG: {} Hz '.format(1/avg_time))