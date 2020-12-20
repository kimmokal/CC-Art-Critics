import torch
from os import listdir, path
from PIL import Image
import torchvision
import pandas as pd


class FeedbackDataset(torch.utils.data.Dataset):

    def __init__(self):
        super(FeedbackDataset, self).__init__()
        currentDir = path.dirname(__file__)
        imageDir = path.join(currentDir, 'image_data/abstract')
        self.imageFiles = [path.join(imageDir, f) for f in listdir(
            imageDir) if path.isfile(path.join(imageDir, f))]
        self.imageNames = [f for f in listdir(
            imageDir) if path.isfile(path.join(imageDir, f))]
        self.ratings = pd.read_csv("art_ratings.csv")

    def __len__(self):
        return len(self.imageFiles)

    def __getitem__(self, index):

        imageName = self.imageNames[index]
        filename = self.imageFiles[index]

        pilImage = Image.open(filename).convert("RGB")
        return (torchvision.transforms.ToTensor()(pilImage), self.ratings[self.ratings["image"] == imageName]["rating"].values[0])
