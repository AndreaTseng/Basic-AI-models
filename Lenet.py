# In this project, we will train a convolutional neural network (CNN) called LeNet
# Understand and count the number of trainable parameters in CNN
# Explore different training configurations such as batch size, learning rate and training epochs.
# Design and customize my own deep network for scene recognition

# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms


class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()
        # certain definitions
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5, stride = 1)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1)
        self.relu = nn.ReLU()
       
    
        self.fc1 = nn.Linear(16 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.printed1 = False
        self.printed2 = False
        self.printed3 = False
        self.printed4 = False
        self.printed5 = False
        self.printed6 = False
        self.printed7 = False
        self.printed8 = False
        self.printed9 = False
        self.printed10 = False
        self.printed11 = False


        #imput layer is 16 channels * (5 x 5) (kernel size is 5)
        #output dimension is 256
        # self.fc1 = nn.Linear(16 * 5 * 5, 256)
        # self.relu3 = nn.ReLU()

        # self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, num_classes)



    def forward(self, x):
        shape_dict = {}
        # certain operations

        

        # if not self.printed1:
        #     print(1, x.shape)  # Example print statement
        #     self.printed1 = True  # Set the flag to True after first print

        x = torch.nn.functional.relu(self.conv1(x))
        # if not self.printed2:
        #     print(2, x.shape)  # Example print statement
        #     self.printed2 = True  # Set the flag to True after first print


        x = self.pool(x)
        # if not self.printed3:
        #     print(3, x.shape)  # Example print statement
        #     self.printed3 = True  # Set the flag to True after first print

        
        shape_dict[1] =list(x.shape)

        x = torch.nn.functional.relu(self.conv2(x))
        # if not self.printed4:
        #     print(4, x.shape)  # Example print statement
        #     self.printed4 = True  # Set the flag to True after first print

        x = self.pool(x)
        # if not self.printed5:
        #     print(5, x)  # Example print statement
        #     self.printed5 = True  # Set the flag to True after first print
        
        shape_dict[2] =list(x.shape)

        x = x.view(-1, 16*5*5)
        # if not self.printed6:
        #     print(6, x.shape)  # Example print statement
        #     self.printed6 = True  # Set the flag to True after first print

        
        shape_dict[3] = list(x.shape)
        x = torch.nn.functional.relu(self.fc1(x))
        # if not self.printed7:
        #     print(7, x.shape)  # Example print statement
        #     self.printed7 = True  # Set the flag to True after first print


        shape_dict[4] = list(x.shape)
        x = torch.nn.functional.relu(self.fc2(x))
        # if not self.printed8:
        #     print(8, x.shape)  # Example print statement
        #     self.printed8 = True  # Set the flag to True after first print

        
        shape_dict[5] = list(x.shape)
        x = self.fc3(x)
        # if not self.printed9:
        #     print(9, x.shape)  # Example print statement
        #     self.printed9 = True  # Set the flag to True after first print

        shape_dict[6] =list(x.shape)
        
        out = x
        #print(shape_dict)
        # if not self.printed10:
        #     print(10, shape_dict)  # Example print statement
        #     self.printed10 = True  # Set the flag to True after first print

        return out, shape_dict


def count_model_params():
    '''
    return the number of trainable parameters of LeNet.
    '''
    model = LeNet()
    total_params = 0  # Initialize total_params


    for parameter in model.parameters():
        if parameter.requires_grad:
            total_params += parameter.numel()

    # Convert to millions
    model_params = total_params / 1e6
    return model_params


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ###################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ###################################

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc
