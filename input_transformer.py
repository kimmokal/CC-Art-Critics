import torch
import numpy as np


class InputTransformer(torch.nn.Module):
    def __init__(self):
        super(InputTransformer, self).__init__()
        self.variances = torch.nn.Parameter(torch.randn(1, 500))
        self.means = torch.nn.Parameter(torch.randn(1, 500))

        self.fc1 = torch.nn.Linear(500, 800)
        self.fc2 = torch.nn.Linear(800, 1000)
        self.noiseFc = torch.nn.Linear(1000, 128)
        self.classFc = torch.nn.Linear(1000, 1000)

    def forward(self, x):
        x = (torch.exp(self.variances)+0.01)*x + self.means
        x = self.fc1(x)
        x = torch.nn.functional.softplus(x) + 0.01
        x = self.fc2(x)
        x = torch.nn.functional.softplus(x) + 0.01
        noise = self.noiseFc(x)
        classes = self.classFc(x)
        classes = torch.sigmoid(classes)
        return noise, classes

    def getNumberOfParameters(self):
        return sum([np.prod(p.size()) for p in self.parameters()])
