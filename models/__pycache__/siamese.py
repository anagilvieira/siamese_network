import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm1d
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
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 1024), #4096 #8192
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
        )
        
        self.classifier = nn.Sequential(
            # First Dense Layer
            nn.Linear(1024, 2),
        )

        """
        # Defining the fully connected layers
        self.fc = nn.Sequential(
        # First Dense Layer
        nn.Linear(512*8, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=0.5),
        # Second Dense Layer
        nn.Linear(1024, 128),
        nn.ReLU(inplace=True),
        # Third Dense Layer
        nn.Linear(128, 2))
        """

    def forward(self, input1, input2):
        output1 = self.features(input1)
        output1 = self.fc(output1)
        # print(output1.shape)
        output2 = self.features(input2)
        output2 = self.fc(output2)
        distance = torch.abs(output1 - output2)
        # print(distance.shape)
        output = self.classifier(distance)
        # print(output)
        return output
