import torch 
import torch.nn as nn
import torch.optim as optim
import imageio
import numpy as np
from utils import *
from torch.utils.tensorboard import SummaryWriter
import matplotlib as plt
from config import *
import tty
import sys
import termios


class NetworkTrainer(object):
    '''
    Class that trains the specified network.
    '''

    def __init__(self, model, dataset, tasks, loss_weights, scaled_anchors, optimizer='sgd', learning_rate=0.0001, momentum=0.9, max_epochs=20, batch_size=2, num_workers=2, device='cuda'):
        super(NetworkTrainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_epochs = max_epochs
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.tasks = tasks
        self.loss_weights = loss_weights
        self.scaled_anchors = scaled_anchors

        self.criterion = {}
        self.criterion['depth'] = nn.MSELoss()
        self.criterion['semantic'] = nn.CrossEntropyLoss()
        self.criterion['bbox'] = YoloLoss().to(device)

        self.input_vis = {}
        self.output_vis_train = {}
        
        

        train_size = int(0.8*len(dataset))
        test_size = len(dataset) - train_size

        self.train_dataset, self.test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        if self.optimizer == 'sgd':
            self.optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)

        self.writer = SummaryWriter()

    def plot_grad_flow(self, named_parameters):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        
        Usage: Plug this function in Trainer class after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads= []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    
    def move_to_device(self, data):
        for k, v in data.items():
                    if torch.is_tensor(data[k]):
                        data[k] = data[k].to(self.device)

                    elif isinstance(data[k], list):
                        #print(data[k])
                        for i in range(len(data[k])):
                            data[k][i] = data[k][i].to(self.device)


    def train(self):
        '''
        Function that trains the network and stores losses and accuracies to be used by tensorboard.
        '''

        train_loss_history = []
        test_loss_history = []

        depth_rmse_history = []


        for epoch in range(self.max_epochs):
            print('>>starting epoch {}/{}<<'.format(epoch+1, self.max_epochs))
            epoch_train_loss = {'total_loss':0.0}
            for t in self.tasks:
                epoch_train_loss[t] = 0.0
            epoch_test_loss = {'total_loss':0.0}
            for t in self.tasks:
                epoch_test_loss[t] = 0.0

            epoch_train_rmse = 0

            for i, data in enumerate(self.train_loader, 0):

                self.move_to_device(data)                       

                self.optimizer.zero_grad()


                outputs = self.model(data)

                loss = 0.0


                for t in self.tasks:
                    task_loss = 0.0
                    
                    if t == 'bbox':

                        #print(data[t][1].shape)

                        new_loss = (
                            self.criterion[t](outputs[t][0], data[t][0], self.scaled_anchors[0])
                            + self.criterion[t](outputs[t][1], data[t][1], self.scaled_anchors[1])
                            + self.criterion[t](outputs[t][2], data[t][2], self.scaled_anchors[2])
                            + self.criterion[t](outputs[t][3], data[t][3], self.scaled_anchors[3])
                        )

                        task_loss = self.loss_weights[t] * new_loss
                        loss += task_loss
                        
                    
                    else:

                        task_loss = self.loss_weights[t] * self.criterion[t](outputs[t], data[t])
                        loss += task_loss

                        # if t == 'depth':
                            
                        #     epoch_train_rmse += self.criterion[t](outputs[t], data[t])
                    
                    epoch_train_loss[t] += task_loss.detach() / self.loss_weights[t]
                    epoch_train_loss['total_loss'] += loss.detach()



                
                
                loss.backward()
                #self.plot_grad_flow(self.model.named_parameters())
                #print(type(torch.autograd.grad))
                
                self.optimizer.step()
                # for tag, parm in self.model.named_parameters():
                #     if parm.grad == None:
                #         print('NONETYPE PARM: {}'.format(tag))
                #     else:
                #         self.writer.add_histogram(tag, parm.grad.clone().cpu().numpy(), epoch)
                

            print('epoch {} train loss: {}'.format(epoch+1, epoch_train_loss))

            for t in self.tasks:
                self.writer.add_scalar('trainig loss for task {}'.format(t), epoch_train_loss[t], epoch)

            self.writer.add_scalar('total trainig loss', epoch_train_loss['total_loss'], epoch)

            if epoch > 0 and epoch % 1 == 0:
            
                for i, data in enumerate(self.test_loader, 0):

                    self.move_to_device(data)

                    outputs = self.model(data)
                    loss = 0.0


                    for t in self.tasks:
                        task_loss = 0.0
                        
                        if t == 'bbox':

                            

                            new_loss = (
                                self.criterion[t](outputs[t][0], data[t][0], self.scaled_anchors[0])
                                + self.criterion[t](outputs[t][1], data[t][1], self.scaled_anchors[1])
                                + self.criterion[t](outputs[t][2], data[t][2], self.scaled_anchors[2])
                                + self.criterion[t](outputs[t][3], data[t][3], self.scaled_anchors[3])
                            )
                            task_loss = self.loss_weights[t] * new_loss
                            loss += task_loss
                        
                        else:
                            task_loss = self.loss_weights[t] * self.criterion[t](outputs[t], data[t])
                            loss += task_loss

                        
                        epoch_test_loss[t] += task_loss.detach() / self.loss_weights[t]
                        epoch_test_loss['total_loss'] += loss.detach()

                
                for t in self.tasks:
                    self.writer.add_scalar('test loss for task {}'.format(t), epoch_test_loss[t], epoch)
                    #self.writer.add_scalar('test loss per task', epoch_test_loss[t], epoch)

                self.writer.add_scalar('total test loss', epoch_test_loss['total_loss'], epoch)

            if epoch > 0 and epoch % 1 == 0:

                self.get_accuracy_metrics(self.test_loader, self.tasks, epoch, 'test')


            #train_loss_history.append(epoch_train_loss)
            

            #depth_rmse_history.append(epoch_train_rmse)

            orig_settings = termios.tcgetattr(sys.stdin)

            tty.setcbreak(sys.stdin)
            if epoch == (self.max_epochs - 1):
                while True:
                    print('Want to save the network? [Y/N]')
                    if sys.stdin.read(1)[0] == 'y':
                        path = os.path.join(os.getcwd(), 'saved_nets/test_network.pt')
                        torch.save(self.model.state_dict(), path)

                        print('saved model to {}'.format(path))
                        break


                    elif sys.stdin.read(1)[0] == 'n':
                        print('network not saved')
                        break

            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig_settings)   


            
    def get_accuracy_metrics(self, loader, tasks, epoch, set):



        for t in tasks:


            if t == 'bbox':
                
                pred_boxes, true_boxes = get_evaluation_bboxes(
                    loader=self.test_loader,
                    model = self.model,
                    iou_threshold=0.5,
                    anchors=cfg.dataset_cfg.anchors,
                    threshold=0.8,
                )
                

                mapval = mean_average_precision(
                    pred_boxes,
                    true_boxes,
                    iou_threshold=0.5,
                    box_format="midpoint",
                    num_classes=cfg.general_cfg.no_classes,
                )
                self.writer.add_scalar('Object detection mAPscore ({}): '.format(set), mapval, epoch)
                print('MAP ACCURACY SCORE OF EPOCH {}: {}'.format(epoch+1, mapval)) 

                idx = 0
                groundtruth = self.dataset[idx]
                
                plot_image_bbox(groundtruth['rgb'][:, :, :].detach().cpu().numpy(), [x[1:] for x in pred_boxes if x[0]==idx], name='pred_test')
                plot_image_bbox(groundtruth['rgb'][:, :, :].detach().cpu().numpy(), [x[1:] for x in true_boxes if x[0]==idx], name='GT_test')


            # if t == 'depth':
            #     depth_RMSE = 0.0
            #     for i, data in enumerate(loader, 0):
            #         self.move_to_device(data)

            #         output = self.model(data)

            #         depth_RMSE += self.criterion[t](output[t], data[t])

            #     self.writer.add_scalar('Depth RMSE ({}): '.format(set), depth_RMSE, epoch)
                

            if t == 'semantic':
                iou_total = 0
                for i, data in enumerate(loader, 0):
                    self.move_to_device(data)

                    output = self.model(data)

                    for j in range(self.batch_size):

                        iou_total += mIOU(data[t][j], output[t][j], cfg.general_cfg.no_classes)

                miou = iou_total / len(loader.dataset)
                self.writer.add_scalar('semantic mIOU ({}): '.format(set), miou, epoch)
                print('semantic mIOU score: {}'.format(miou))


            else:
                print('accuracy metric for task {} not implemented!'.format(t))





    def visualize(self, idx, dataset, vis_tasks, model):
        '''
        Function that gives qualitative results by saving ground truth and output images per task for a given input image.
        '''


        groundtruth = dataset[idx]



        for k, v in groundtruth.items():
            if torch.is_tensor(groundtruth[k]):
                groundtruth[k] = torch.unsqueeze(v, 0).to(self.device)

            elif isinstance(groundtruth[k], list):
                        #print(data[k])
                        for i in range(len(groundtruth[k])):
                            groundtruth[k][i] = groundtruth[k][i].to(self.device)
        output = model(groundtruth)#.to(self.device)
        print(output.keys())

        for t in vis_tasks:
            print('writing images of {} task'.format(t))
            #print(groundtruth[t].shape)
            #print(output[t].shape)



            if t == 'semantic':
                sem_output = torch.argmax(output[t].squeeze(), dim=0).detach().cpu().numpy()
                print(sem_output.shape)
                def decode_segmap(image):
                    label_colors = np.array([(0,0,0), (128, 0, 0), (0, 128, 0), (0, 0, 128)])
                    r = np.zeros_like(sem_output).astype(np.uint8)
                    g = np.zeros_like(sem_output).astype(np.uint8)
                    b = np.zeros_like(sem_output).astype(np.uint8)
                    for l in range(0, 3):
                        idx = image == l
                        r[idx] = label_colors[l, 0]
                        g[idx] = label_colors[l, 1]
                        b[idx] = label_colors[l, 2]
                    rgb = np.stack([r, g, b], axis=2)

                    return rgb

                imageio.imwrite('testing/{}_GT.jpeg'.format(t), decode_segmap(groundtruth[t][0, :, :].cpu()))
                imageio.imwrite('testing/{}_output.jpeg'.format(t), decode_segmap(sem_output))

            elif t == 'bbox':
                print('old input image shape: {}'.format(groundtruth['rgb'].shape))
                #groundtruth['rgb'] = torch.from_numpy(imageio.imread('warehouse.png')).unsqueeze(0).unsqueeze(0).type(torch.float32).to(device) 
                print(groundtruth['rgb'].shape)   
                #output = model(groundtruth)
                print(type(output))
                pred_bboxes = []
                gt_bboxes = []

                # S2 = cfg.dataset_cfg.S.copy()
                # S=cfg.dataset_cfg.S.copy()
                # S.reverse()

                # print(S)
                # print(S2)

                for i in range(len(output[t])):
                    #print(type(output[t]))
                    # print(output[t][i].shape)
                    # print(torch.tensor(cfg.dataset_cfg.anchors).to(self.device)[i])
                    # print(S[i])
                    print('gt shape before c2b: {}'.format(groundtruth[t][i].shape))

                    pred_bboxes.extend((cells_to_bboxes(output[t][i], self.scaled_anchors[i], S=output[t][i].shape[2])))
                    gt_bboxes.extend((cells_to_bboxes(groundtruth[t][i].unsqueeze(0), self.scaled_anchors[i], S=groundtruth[t][i].unsqueeze(0).shape[2], is_preds=False)))

                pred_bboxes_flat = [item for sublist in pred_bboxes for item in sublist]
                gt_bboxes_flat = [item for sublist in gt_bboxes for item in sublist]


                print('length of bbox list: {}'.format(len(pred_bboxes_flat)))
                print(pred_bboxes_flat[0])
                pred_bboxes_nms = non_max_suppression(pred_bboxes_flat, 0.5, 0.8)
                print('length pred bbox list nms: {}'.format(len(pred_bboxes_nms)))
                gt_bboxes_nms = non_max_suppression(gt_bboxes_flat, .9, .9)
                print('length  gt bbox list flat: {}'.format(len(gt_bboxes_flat)))

                print('rgb shape: {}'.format(groundtruth['rgb'][0, :, :].detach().cpu().numpy().shape))

                plot_image_bbox(groundtruth['rgb'][0, :, :].detach().cpu().numpy(), pred_bboxes_nms)
                plot_image_bbox(groundtruth['rgb'][0, :, :].detach().cpu().numpy(), gt_bboxes_nms, name='GT')

                




            else:
                imageio.imwrite('testing/{}_GT.jpeg'.format(t), groundtruth[t][0, :, :].detach().cpu().numpy())
                imageio.imwrite('testing/{}_output.jpeg'.format(t), output[t][0, 0, :, :].detach().cpu().numpy())



    
    

















