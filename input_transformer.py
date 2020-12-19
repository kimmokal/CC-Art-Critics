import torch
import numpy as np


class InputTransformer(torch.nn.Module):
    def __init__(self):
        super(InputTransformer, self).__init__()
        self.variances = torch.nn.Parameter(torch.randn(1, 500))
        self.means = torch.nn.Parameter(torch.randn(1, 500))

        self.noiseFc = torch.nn.Linear(500, 128, bias=False)
        self.classFc = torch.nn.Linear(500, 1000, bias=False)

    def forward(self, x):
        x = (torch.exp(self.variances)+1)*x + self.means
        noise = self.noiseFc(x)
        classes = self.classFc(x)
        classes = torch.nn.functional.softmax(classes, dim=1)
        return noise, classes

    def getNumberOfParameters(self):
        return sum([np.prod(p.size()) for p in self.parameters()])
