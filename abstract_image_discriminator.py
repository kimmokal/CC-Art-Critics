import torch
import numpy as np


class AbstractImageDiscriminator(torch.nn.Module):
    def __init__(self):
        super(AbstractImageDiscriminator, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 11, stride=5, dilation=1)
        self.conv2 = torch.nn.Conv2d(64, 128, 7, stride=3, dilation=1)
        self.conv3 = torch.nn.Conv2d(128, 256, 5, stride=1, dilation=1)
        self.fc = torch.nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.conv3(x)
        x = torch.nn.functional.max_pool2d(x, x.shape[2:])
        x = torch.nn.functional.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

    def getNumberOfParameters(self):
        return sum([np.prod(p.size()) for p in self.parameters()])
