import torch.nn as nn
from . import utils
import torch


@utils.register_model
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Flatten(),
            nn.Linear(8192, 1024), #4096 #8192
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
        )
        
        self.classifier = nn.Sequential(
            # First Dense Layer
            nn.Linear(1024, 2),
        )
    

    def forward(self, input1, input2):
        output1 = self.features(input1.float())
        # print(output1.shape)
        output2 = self.features(input2.float())
        distance = torch.abs(output1 - output2)
        # print(distance.shape)
        output = self.classifier(distance)
        # print(output)
        return output
