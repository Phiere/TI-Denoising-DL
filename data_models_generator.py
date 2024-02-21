
import glob
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
from torch.utils.data import Dataset
import random


mean = [0.4361, 0.4435, 0.4457]
std = [0.1523, 0.1478, 0.1473]

class DenoisingDataset(Dataset):
    """Dataset wrapping tensors.
    """
    def __init__(self,data_path,sigma,patch_size = 35,training = True):
        super(DenoisingDataset, self).__init__()
        self.file_list = glob.glob(data_path+'/*.png')

        self.transforms = torchvision.transforms.Compose([#torchvision.transforms.Normalize(mean = mean,std = std),                                                      
                                                          torchvision.transforms.RandomResizedCrop(size =patch_size,antialias=None),
                                                          torchvision.transforms.RandomHorizontalFlip(),
                                                          torchvision.transforms.RandomVerticalFlip(),
                                                          torchvision.transforms.RandomRotation(30)
                                                          ])
        self.sigma = sigma
        self.training = training

    def __getitem__(self, index):
        batch_x = torchvision.io.read_image(self.file_list[index])
        batch_x = batch_x.float()/255.0

        if self.training : batch_x = self.transforms(batch_x)
        if self.sigma == -1 : self.sigma = random.randint(0,50)
        noise = torch.randn(batch_x.size()).mul_(self.sigma/255.0)
        batch_y = batch_x + noise
        return batch_y, batch_x, noise

    def __len__(self):
        return len(self.file_list)


class PersonalDenoiserFataset(Dataset):
    """Dataset pour débruiter des images déjà bruitée. Seulement utile pour une mise en application.
    """
    def __init__(self,data_path):
        super(PersonalDenoiserFataset, self).__init__()
        self.file_list = self.file_list = glob.glob(data_path+'/*.png')

    def __getitem__(self, index):
        batch_x = torchvision.io.read_image(self.file_list[index])
        batch_x = batch_x.float()/255.0

        return batch_x

    def __len__(self):
        return len(self.file_list)


##Va falloir essayer de comprendre les paramètres genre momentum ?
class DnCNN(nn.Module):
    def __init__(self, depth=7, n_channels=64, image_channels=3, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        out = self.dncnn(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

