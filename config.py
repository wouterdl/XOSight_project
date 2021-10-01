from easydict import EasyDict as edict
import os, os.path
from network_modules import *
import network_modules

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##General

general_cfg = edict(loss_weights={})


general_cfg.loss_weights['depth'] = 1
general_cfg.loss_weights['semantic'] = 1
general_cfg.loss_weights['bbox'] = 0.01

general_cfg.no_classes = 3
general_cfg.img_size = 256

general_cfg.TASKS = edict(NAMES=['depth', 'semantic', 'bbox'])

##Dataset

dataset_cfg = edict()
dataset_cfg['data_path'] = os.path.join(os.getcwd(), 'output/Viewport')
dataset_cfg['anchors'] = [
        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
        ]

dataset_cfg['S'] = [8, 16, 32, 64]
dataset_cfg['iou_thresh'] = 0.5


##HRnet backbone

HRnet_cfg = edict()
HRnet_cfg['STAGE1'], HRnet_cfg['STAGE2'], HRnet_cfg['STAGE3'], HRnet_cfg['STAGE4'] = edict(), edict(), edict(), edict()


HRnet_cfg['STAGE1']['NUM_MODULES'] = 1
HRnet_cfg['STAGE1']['NUM_BRANCHES'] = 1
HRnet_cfg['STAGE1']['NUM_BLOCKS'] = [4]
HRnet_cfg['STAGE1']['NUM_CHANNELS'] = [64]
HRnet_cfg['STAGE1']['BLOCK'] = 'BOTTLENECK'
HRnet_cfg['STAGE1']['FUSE_METHOD'] = 'SUM'

HRnet_cfg['STAGE2']['NUM_MODULES'] = 1
HRnet_cfg['STAGE2']['NUM_BRANCHES'] = 2
HRnet_cfg['STAGE2']['NUM_BLOCKS'] = [4, 4]
HRnet_cfg['STAGE2']['NUM_CHANNELS'] = [18, 36]
HRnet_cfg['STAGE2']['BLOCK'] = 'BASIC'
HRnet_cfg['STAGE2']['FUSE_METHOD'] = 'SUM'

HRnet_cfg['STAGE3']['NUM_MODULES'] = 4
HRnet_cfg['STAGE3']['NUM_BRANCHES'] = 3
HRnet_cfg['STAGE3']['NUM_BLOCKS'] = [4, 4, 4]
HRnet_cfg['STAGE3']['NUM_CHANNELS'] = [18, 36, 72]
HRnet_cfg['STAGE3']['BLOCK'] = 'BASIC'
HRnet_cfg['STAGE3']['FUSE_METHOD'] = 'SUM'

HRnet_cfg['STAGE4']['NUM_MODULES'] = 3
HRnet_cfg['STAGE4']['NUM_BRANCHES'] = 4
HRnet_cfg['STAGE4']['NUM_BLOCKS'] = [4, 4, 4, 4]
HRnet_cfg['STAGE4']['NUM_CHANNELS'] = [18, 36, 72, 144]
HRnet_cfg['STAGE4']['BLOCK'] = 'BASIC'
HRnet_cfg['STAGE4']['FUSE_METHOD'] = 'SUM'

##MTI-Net

MTI_cfg = edict(heads={})

MTI_cfg.TASKS = edict(NAMES=['depth', 'semantic'])
MTI_cfg.AUXILARY_TASKS = edict(NAMES=['depth', 'semantic'])


MTI_cfg.AUXILARY_TASKS.NUM_OUTPUT = edict(depth=1, semantic=general_cfg.no_classes)


MTI_cfg.backbone_channels=[18, 36, 72, 144]
MTI_cfg.img_size = general_cfg.img_size
MTI_cfg.heads.depth = network_modules.HighResolutionHead(backbone_channels=MTI_cfg.backbone_channels, num_outputs=1).to(device)
MTI_cfg.heads.semantic = network_modules.HighResolutionHead(backbone_channels=MTI_cfg.backbone_channels, num_outputs=general_cfg.no_classes).to(device)

##YOLO head

yolo_cfg = edict()
yolo_cfg.num_classes = general_cfg.no_classes
yolo_cfg.anchors_per_scale = 3


#Combine toghether

cfg = edict(dataset_cfg=dataset_cfg, HRnet_cfg=HRnet_cfg, MTI_cfg=MTI_cfg, general_cfg=general_cfg, yolo_cfg=yolo_cfg)
