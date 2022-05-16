import torch.nn as nn
from . import utils
import torch
import numpy as np


@utils.register_model
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2, stride=2)
        self.batch1 = nn.BatchNorm2d(64)
        
        self.relu = nn.ReLU(inplace=True)
        self.max = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=2)
        self.batch2 = nn.BatchNorm2d(128)


        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.batch3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)
        self.batch4 = nn.BatchNorm2d(512)
        
        #self.gap = nn.AvgPool2d((1,1)) 
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8192, 1024) #4096 #8192
        self.fc2 = nn.Linear(1024, 2)
    

    def forward(self, input1, input2):
        
        output1 = self.conv1(input1.float())
        output1 = self.relu(output1)
        output1 = self.batch1(output1)
        output1 = self.max(output1)
        
        output1 = self.conv2(output1)
        output1 = self.relu(output1)
        output1 = self.batch2(output1)
        output1 = self.max(output1)
        
        output1 = self.conv3(output1)
        output1 = self.relu(output1)
        output1 = self.batch3(output1)
        output1 = self.max(output1)
        
        output1 = self.conv4(output1)
        output1 = self.relu(output1)
        output1 = self.batch4(output1)  # 512*4*4
        
        output1 = self.flatten(output1)
        
        
        output2 = self.conv1(input2.float())
        output2 = self.relu(output2)
        output2 = self.batch1(output2)
        output2 = self.max(output2)
        
        output2 = self.conv2(output2)
        output2 = self.relu(output2)
        output2 = self.batch2(output2)
        output2 = self.max(output2)
        
        output2 = self.conv3(output2)
        output2 = self.relu(output2)
        output2 = self.batch3(output2)
        output2 = self.max(output2)
        
        output2 = self.conv4(output2)
        output2 = self.relu(output2)
        output2 = self.batch4(output2)
        
        output2 = self.flatten(output2)
        

        distance = torch.abs(output1 - output2)
        # print(distance.shape)
        
        output = self.fc1(distance)
        output = self.fc2(output)
        # print(output)
        return output
