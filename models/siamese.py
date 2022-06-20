import torch.nn as nn
from . import utils
import torch
import copy


@utils.register_model
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        #self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2, stride=2)
        #self.batch1 = nn.BatchNorm2d(32)
        
        self.relu = nn.ReLU(inplace=True)
        self.max = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(1, 64, kernel_size=5, padding=2, stride=2)
        self.batch2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=2)
        self.batch3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.batch4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)
        self.batch5 = nn.BatchNorm2d(512)
        
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(8192, 1024)
        self.fc2 = nn.Linear(1024, 2)   


        self.encoder = nn.Sequential(
            
            self.conv2, self.relu, self.batch2, self.max,
            self.conv3, self.relu, self.batch3, self.max,
            self.conv4, self.relu, self.batch4, self.max,
            self.conv5, self.relu, self.batch5, self.flatten)

        self.encoder2 = copy.deepcopy(self.encoder)


    def forward(self, input1, input2):
        
        output1 = self.encoder(input1.float())
        output2 = self.encoder2(input2.float())

        distance = torch.abs(output1 - output2)
        #output = self.avgpool(distance)
        #output = output.view(output.size(0), -1)
        
        output = self.fc1(distance)
        output = self.fc2(output)
        
        return output