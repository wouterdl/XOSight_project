import numpy as np
import torch
import torch.nn as nn
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from collections import Counter
from config import *




def mIOU(gt, output, n_classes):
    #gt: [H * W]
    #output: [C * H * W]

    #H, W = output.shape[1], output.shape[2]

    #results = np.zeros((H, W))
    #print('gt shape: {}'.format(gt.shape))
    #print('output shape: {}'.format(output.shape))

    output = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()

    #print('gt shape: {}'.format(gt.shape))
    #print('output shape: {}'.format(output.shape))

    intersection = np.logical_and(gt, output)
    union = np.logical_or(gt, output)
    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score


    

def iou_width_height(boxes1, boxes2):
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Video explanation of this function:
    https://youtu.be/XXYG5ZWtjj0
    This function calculates intersection over union (iou) given pred boxes
    and target boxes.
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)
    
        

class YoloLoss(nn.Module):
    '''
    Class that calculates the loss of the YOLO head.
    '''
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 1
        self.lambda_obj = 1
        self.lambda_box = 1

    def forward(self, predictions, target, anchors):
        """
        :param predictions: output from model of shape: (batch size, anchors on scale , grid size, grid size, 5 + num classes)
        :param target: targets on particular scale of shape: (batch size, anchors on scale, grid size, grid size, 6)
        :param anchors: anchor boxes on the particular scale of shape (anchors on scale, 2)
        :return: returns the loss on the particular scale
        """
        # print('target shape: {}'.format(target.is_cuda))
        # print('prediction shape: {}'.format(predictions.is_cuda))
        # Check where obj and noobj (we ignore if target == -1)
        # Here we check where in the label matrix there is an object or not
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #
        # The indexing noobj refers to the fact that we only apply the loss where there is no object
        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #
        # Here we compute the loss for the cells and anchor boxes that contain an object
        # Reschape anchors to allow for broadcasting in multiplication below
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        # Convert outputs from model to bounding boxes according to formulas in paper
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        # Targets for the object prediction should be the iou of the predicted bounding box and the target bounding box
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        # Only incur loss for the cells where there is an objects signified by indexing with obj
        object_loss = self.bce((predictions[..., 0:1][obj]), (ious * target[..., 0:1][obj]))

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #
        # apply sigmoid to x, y coordinates to convert to bounding boxes
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3]) 
        # to improve gradient flow we convert targets' width and height to the same format as predictions
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        ) 
        # compute mse loss for boxes
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #
        # here we just apply cross entropy loss as is customary with classification problems
        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()),
        )
        
        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )  



def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    """
    Scales the predictions coming from the model to
    be relative to the entire image such that they for example later
    can be plotted or.
    INPUT:
    predictions: tensor of size (N, 3, S, S, num_classes+5)
    anchors: the anchors used for the predictions
    S: the number of cells the image is divided in on the width (and height)
    is_preds: whether the input is predictions or the true bounding boxes
    OUTPUT:
    converted_bboxes: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
                      object score, bounding box coordinates
    """

    


    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    print(box_predictions[..., 0:1].shape)
    print(cell_indices.shape)
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
    return converted_bboxes.tolist()

# def non_max_suppression(bboxes, iou_threshold, threshold, box_format="midpoint"):
#     """
#     Video explanation of this function:
#     https://youtu.be/YDkjWEN8jNA
#     Does Non Max Suppression given bboxes
#     Parameters:
#         bboxes (list): list of lists containing all bboxes with each bboxes
#         specified as [class_pred, prob_score, x1, y1, x2, y2]
#         iou_threshold (float): threshold where predicted bboxes is correct
#         threshold (float): threshold to remove predicted bboxes (independent of IoU)
#         box_format (str): "midpoint" or "corners" used to specify bboxes
#     Returns:
#         list: bboxes after performing NMS given a specific IoU threshold
#     """

#     assert type(bboxes) == list
#     idx_bool = False
#     if len(bboxes[0]) == 7:
#         idx_bool = True
    
#     if idx_bool == True:
#         bboxes = [box for box in bboxes if box[2] > threshold]
#         bboxes = sorted(bboxes, key=lambda x: x[2], reverse=True)
#     else:
#         bboxes = [box for box in bboxes if box[1] > threshold]
#         bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)


    
#     bboxes_after_nms = []

#     while bboxes:
#         chosen_box = bboxes.pop(0)
#         if idx_bool == True:
#             idx = chosen_box[0] 

#             bboxes = [
#                 box
#                 for box in bboxes
#                 if box[1] != chosen_box[1]
#                 or intersection_over_union(
#                     torch.tensor(chosen_box[3:]),
#                     torch.tensor(box[3:]),
#                     box_format=box_format,
#                 )
#                 < iou_threshold
#             ]
#             bboxes_after_nms.append([idx] + chosen_box)

#         else:
#             bboxes = [
#                 box
#                 for box in bboxes
#                 if box[0] != chosen_box[0]
#                 or intersection_over_union(
#                     torch.tensor(chosen_box[2:]),
#                     torch.tensor(box[2:]),
#                     box_format=box_format,
#                 )
#                 < iou_threshold
#             ]

#             bboxes_after_nms.append(chosen_box)


#     return bboxes_after_nms

def non_max_suppression(bboxes, iou_threshold, threshold, box_format="midpoint"):
    """
    Video explanation of this function:
    https://youtu.be/YDkjWEN8jNA
    Does Non Max Suppression given bboxes
    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes
    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms



def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=3
):
    """
    Video explanation of this function:
    https://youtu.be/FppOzcDvaDI
    This function calculates mean average precision (mAP)
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6
   
    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)
                #detections.append(true_box)
         
        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )
                   
                if iou > best_iou:
                    
                    best_iou = iou
                    best_gt_idx = idx
                      
            
            
            if best_iou > iou_threshold:
                
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    print('another TP for class {}'.format(c))
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        
        

        average_precisions.append(torch.trapz(precisions, recalls))
        print('AP for class {}: {}'.format(c, torch.trapz(precisions, recalls)))

    return sum(average_precisions) / len(average_precisions)


# def plot_image_bbox(image, boxes, name='output'):
#     """Plots predicted bounding boxes on the image"""
#     #cmap = plt.get_cmap("tab20b")
#     #class_labels = config.COCO_LABELS if config.DATASET=='COCO' else config.PASCAL_CLASSES
#     #colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
#     class_labels = [0, 1, 2]
#     colors = [(128, 0, 0), (0, 128, 0), (0, 0, 128)]

#     im = np.array(image).transpose(1, 2, 0)
#     height, width, _ = im.shape
#     print(im.shape)
#     # Create figure and axes
#     # fig, ax = plt.subplots(1)
#     # # Display the image
#     # ax.imshow(im)

#     # box[0] is x midpoint, box[2] is width
#     # box[1] is y midpoint, box[3] is height

#     # Create a Rectangle patch

#     #image = cv.fromarray(image)
#     image =  np.ascontiguousarray(image, dtype=np.uint8)
#     for box in boxes:
#         assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
#         class_pred = box[0]
#         box = box[2:]
#         upper_left_x = box[0] - box[2] / 2
#         upper_left_y = box[1] - box[3] / 2
#         # rect = patches.Rectangle(
#         #     (upper_left_x * width, upper_left_y * height),
#         #     box[2] * width,
#         #     box[3] * height,
#         #     linewidth=2,
#         #     edgecolor=colors[int(class_pred)],
#         #     facecolor="none",
#         # )
#         # print(upper_left_x)
#         # print(upper_left_y)
#         # print(upper_left_x + box[0])
#         # print(upper_left_y + box[1])

#         print(box[0])
#         print(box[1])

#         x1 = int(upper_left_x * width)
#         y1 = int(upper_left_y * height)
#         x2 = int((upper_left_x + box[0]) * width)
#         y2 = int((upper_left_y + box[1]) * height)
#         print(x1)
#         print(y1)
#         print(x2)
#         print(y2)

#         #cv2.rectangle(image, (upper_left_x, upper_left_y), (upper_left_x + box[0], upper_left_y + box[1]), color=colors[int(class_pred)], thickness=2)
#         cv2.rectangle(image, (x1, y1), (x2, y2), (128, 0, 0), 2)

#         # Add the patch to the Axes
#         # ax.add_patch(rect)
#         # plt.text(
#         #     upper_left_x * width,
#         #     upper_left_y * height,
#         #     s=class_labels[int(class_pred)],
#         #     color="white",
#         #     verticalalignment="top",
#         #     bbox={"color": colors[int(class_pred)], "pad": 0},
#         # )
#     print(image.shape)
#     cv2.imwrite('bbox_{}.jpeg'.format(name), image)
#     #plt.show()


def plot_image_bbox(image, boxes, name='output'):
    """Plots predicted bounding boxes on the image"""
    #cmap = plt.get_cmap("tab20b")
    #class_labels = config.COCO_LABELS if config.DATASET=='COCO' else config.PASCAL_CLASSES
    #colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    class_labels = [0, 1, 2, 3]
    colors = [(128/255, 0, 0), (0, 128/255, 0), (0, 0, 128/255), (128/255, 128/255, 128/255)]

    im = np.array(image).transpose(1, 2, 0)/255
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    #ax.imshow(im, cmap='gray')
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=class_labels[int(class_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )

    plt.savefig('bbox_{}'.format(name))

def get_evaluation_bboxes(
    loader,
    model,
    iou_threshold,
    anchors,
    threshold,
    box_format="midpoint",
    device="cuda",
):

    scaled_anchors = (
        torch.tensor(cfg.dataset_cfg.anchors)
        * torch.tensor(cfg.dataset_cfg.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(device)

    
    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, data in enumerate(tqdm(loader)):
        
        x = data
        for k, v in x.items():
                    if torch.is_tensor(x[k]):
                        x[k] = x[k].to(device)

                    elif isinstance(x[k], list):
                        
                        for i in range(len(x[k])):
                            x[k][i] = x[k][i].to(device)

        labels = data['bbox']

        with torch.no_grad():
            predictions = model(x)
        predictions = predictions['bbox']
        print('predictions shape: {}'.format(predictions[2].shape))
        print('labels shape: {}'.format(labels[2].shape))
        batch_size = x['rgb'].shape[0]
        
        bboxes = [[] for _ in range(batch_size)]
        true_bboxes = [[] for _ in range(batch_size)]
        for i in range(4):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device) * S
            
            boxes_scale_i = cells_to_bboxes(
                predictions[i], scaled_anchors[i], S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box
            

        #we just want one bbox for each label, not one for each scale
        true_bboxes = cells_to_bboxes(
            labels[3], anchor, S=S, is_preds=False
        )

        for idx in range(batch_size):

            print('BBOX pre-NMS: {}'.format(len(bboxes[idx])))
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            
            for nms_box in nms_boxes:

                all_pred_boxes.append([train_idx] + nms_box)
            

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()

    return all_pred_boxes, all_true_boxes