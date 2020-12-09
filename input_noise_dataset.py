import torch


class NoiseDataset(torch.utils.data.Dataset):

    def __init__(self, size=1000):
        super(NoiseDataset, self).__init__()
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return torch.randn((500,)), 1
