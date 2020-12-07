import torch
from os import listdir, path
from PIL import Image
import torchvision


class DiscriminatorDataset(torch.utils.data.Dataset):

    def __init__(self):
        super(DiscriminatorDataset, self).__init__()
        currentDir = path.dirname(__file__)
        abstractDir = path.join(currentDir, 'image_data/abstract')
        realisticDir = path.join(currentDir, 'image_data/realistic')
        abstractFiles = [path.join(abstractDir, f) for f in listdir(
            abstractDir) if path.isfile(path.join(abstractDir, f))]
        realisticFiles = [path.join(realisticDir, f) for f in listdir(
            realisticDir) if path.isfile(path.join(realisticDir, f))]
        self.abstractFilesLen = len(abstractFiles)
        self.allFiles = abstractFiles + realisticFiles

    def __len__(self):
        return len(self.allFiles)

    def __getitem__(self, index):
        filename = self.allFiles[index]
        pilImage = Image.open(filename).convert("RGB")
        return (torchvision.transforms.ToTensor()(pilImage), 1 if index < self.abstractFilesLen else 0)
