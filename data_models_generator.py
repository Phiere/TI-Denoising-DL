
import glob
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
from torch.utils.data import Dataset
import random
import os 
import numpy as np

#Verifier que c'est bien rgb et pas autre chose
#Au lieu de mettre training en argument, mettre les transformations ?

mean_db = [0.4273080293308286, 0.4416487863396897, 0.3839654446963002] 
std_db = [0.21955110505223274, 0.20601368640713832, 0.20300565298427553]


class DenoisingDataset(Dataset):
    """Dataset wrapping tensors.
    """
    def __init__(self,data_path,sigma,patch_size = 35,training = True):
        super(DenoisingDataset, self).__init__()
        self.file_list = glob.glob(data_path+'/*.png')
      
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean = mean_db,std = std_db),                                                      
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

        if self.training : 
            batch_x = self.transforms(batch_x)
        if self.sigma == -1 : 
            self.sigma = random.randint(0,50)

        noise = torch.randn(batch_x.size()).mul_(self.sigma/255.0)
        batch_y = batch_x + noise
        return batch_y, batch_x, noise

    def __len__(self):
        return len(self.file_list)


class PersonalDenoiserFataset(Dataset):
    """Dataset pour débruiter des images déjà bruitées. Seulement utile pour une mise en application.
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
    """Création du réseau de neurones."""
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
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

def mean_std_db_calcul(images_dir):
    """Calculs des means et std de la data base d'entraînement (canal par canal)"""
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_file_directory,images_dir)
    file_list = glob.glob(data_path+'/*.png')
    
    means_R,means_G,means_B = [],[],[]
    std_R,std_G,std_B = [],[],[]


    for file in file_list :
        image = torchvision.io.read_image(file)
        image = image.float()/255.0

        image_R = np.array(image[0,:,:])
        image_G = np.array(image[1,:,:])
        image_B = np.array(image[2,:,:])

        means_R.append(np.mean(image_R))
        means_G.append(np.mean(image_G))
        means_B.append(np.mean(image_B))

        std_R.append(np.std(image_R))
        std_G.append(np.std(image_G))
        std_B.append(np.std(image_B))
    
    def mean_l(tab):
        return sum(tab)/len(tab)
    
    means = [mean_l(means_R),mean_l(means_G),mean_l(means_B)]
    stds = [mean_l(std_R),mean_l(std_G),mean_l(std_B)]
    
    return means, stds


