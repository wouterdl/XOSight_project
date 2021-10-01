import torch 
import torch.nn as nn
import torch.optim as optim
import imageio
import numpy as np
from utils import *
from torch.utils.tensorboard import SummaryWriter
import matplotlib as plt


class NetworkTrainer(object):

    def __init__(self, model, dataset, tasks, loss_weights, scaled_anchors, optimizer='sgd', learning_rate=0.005, momentum=0.9, max_epochs=20, batch_size=2, num_workers=2, device='cuda'):
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

    
    
    def train(self):

        train_loss_history = []
        test_loss_history = []

        depth_rmse_history = []


        for epoch in range(self.max_epochs):
            print('>>starting epoch {}/{}<<'.format(epoch+1, self.max_epochs))
            epoch_train_loss = {'total_loss':0.0}
            for t in self.tasks:
                epoch_train_loss[t] = 0.0
            epoch_test_loss = 0.0

            epoch_train_rmse = 0

            for i, data in enumerate(self.train_loader, 0):

                for k, v in data.items():
                    if torch.is_tensor(data[k]):
                        data[k] = data[k].to(self.device)

                    elif isinstance(data[k], list):
                        #print(data[k])
                        for i in range(len(data[k])):
                            data[k][i] = data[k][i].to(self.device)
                       

                self.optimizer.zero_grad()

                outputs = self.model(data)
                loss = 0.0


                for t in self.tasks:
                    
                    
                    if t == 'bbox':

                        #print(data[t][1].shape)

                        new_loss = (
                            self.criterion[t](outputs[t][0], data[t][3], self.scaled_anchors[0])
                            + self.criterion[t](outputs[t][1], data[t][2], self.scaled_anchors[1])
                            + self.criterion[t](outputs[t][2], data[t][1], self.scaled_anchors[2])
                            + self.criterion[t](outputs[t][3], data[t][0], self.scaled_anchors[3])
                        )

                        loss += self.loss_weights[t] * new_loss
                    
                    else:
                        loss += self.loss_weights[t] * self.criterion[t](outputs[t], data[t])

                        if t == 'depth':
                            
                            epoch_train_rmse += self.criterion[t](outputs[t], data[t])
                    
                    epoch_train_loss[t] += loss / self.loss_weights[t]



                
                epoch_train_loss['total_loss'] += loss
                loss.backward()
                #self.plot_grad_flow(self.model.named_parameters())
                self.optimizer.step()
                # for tag, parm in self.model.named_parameters():
                #     self.writer.add_histogram(tag, parm.grad.data.clone().cpu().numpy(), epoch)

                

            print('epoch {} train loss: {}'.format(epoch+1, epoch_train_loss))

            for t in self.tasks:
                self.writer.add_scalar('trainig loss for task {}'.format(t), epoch_train_loss[t], epoch)

            self.writer.add_scalar('total trainig loss', epoch_train_loss['total_loss'], epoch)



            # for i, data in enumerate(self.test_loader, 0):

            #     for k, v in data.items():
            #         data[k] = data[k].to(self.device)

            #     outputs = self.model(data)
            #     loss = 0.0


            #     for t in self.tasks:
            #         #print('calculating loss for task: {}'.format(t))
            #         # if t == 'semantic':
            #         #     outputs[t] = torch.argmax(outputs[t].squeeze(), dim=1).type(torch.int64)
            #         #     #print(outputs[t])
            #         #print(outputs[t].type())    
            #         loss += self.loss_weights[t] * self.criterion[t](torch.squeeze(outputs[t]), data[t])



                
            #     epoch_test_loss += loss

            # print('epoch {} test loss: {}'.format(epoch+1, epoch_test_loss))

            train_loss_history.append(epoch_train_loss)
            

            depth_rmse_history.append(epoch_train_rmse)


            
            



    def visualize(self, idx, dataset, vis_tasks, model):

        groundtruth = dataset[idx]



        for k, v in groundtruth.items():
            groundtruth[k] = torch.unsqueeze(v, 0).to(self.device)
        output = model(groundtruth)#.to(self.device)

        for t in vis_tasks:
            print('writing images of {} task'.format(t))
            #print(groundtruth[t].shape)
            #print(output[t].shape)



            if t == 'semantic':
                output = torch.argmax(output[t].squeeze(), dim=0).detach().cpu().numpy()
                print(output.shape)
                def decode_segmap(image):
                    label_colors = np.array([(0,0,0), (128, 0, 0), (0, 128, 0), (0, 0, 128)])
                    r = np.zeros_like(output).astype(np.uint8)
                    g = np.zeros_like(output).astype(np.uint8)
                    b = np.zeros_like(output).astype(np.uint8)
                    for l in range(0, 3):
                        idx = image == l
                        r[idx] = label_colors[l, 0]
                        g[idx] = label_colors[l, 1]
                        b[idx] = label_colors[l, 2]
                    rgb = np.stack([r, g, b], axis=2)

                    return rgb

                imageio.imwrite('testing/{}_GT.jpeg'.format(t), decode_segmap(groundtruth[t][0, :, :].cpu()))
                imageio.imwrite('testing/{}_output.jpeg'.format(t), decode_segmap(output))
                
            else:
                imageio.imwrite('testing/{}_GT.jpeg'.format(t), groundtruth[t][0, :, :].detach().cpu().numpy())
                imageio.imwrite('testing/{}_output.jpeg'.format(t), output[t][0, 0, :, :].detach().cpu().numpy())
    
    

















