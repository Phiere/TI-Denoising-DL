import data_models_generator as dmg
import torch.optim as optim
import torch,os
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description='PyTorch DnCNN')
parser.add_argument('--model_name', default='model_debruiteur.pth', type=str, help="Nom du model qu'on souhaite entrainer ou tester")
parser.add_argument('--sigma', default=25, type=int, help="Choix du sigma à utliser pour ,l'entrainement et le test. Attention un sigma de -1 rendra le sigma alétoire entre 0 et 50")
parser.add_argument('--image_channels', default=3,type=int, help="permet de choisir si les images traitée sont en rgb ou en niveau de gris")

##Données d'entrainement
parser.add_argument('--train_data', default='/Data_gray/Train', type=str, help='path of train data')
#Paramètre de sauvegarde
parser.add_argument('--model_dir', default='/Models_gray', help='directory of the model')
parser.add_argument('--loss_dir', default='/losses/loss', help='directory of the loss plot')
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
args, unknown = parser.parse_known_args()

current_file_directory = os.path.dirname(os.path.abspath(__file__))
###Choix des hyper_paramètres
training_batch_size = 1
training_nb_epoch = 1000
training_learning_rate = 1e-3
training_patch_size = 35

training_network = dmg.DnCNN(image_channels=args.image_channels)
training_optimizer = optim.Adam(training_network.parameters(), lr=args.lr)
training_criterion = F.mse_loss
training_scheduler = MultiStepLR(training_optimizer, milestones=[training_nb_epoch/2], gamma=0.1) 


def model_training(network,optimizer,criterion,scheduler,nb_epoch,patch_size,batch_size) :
    # model selection
    print('===> Building model')

    model = network
    DDataset = dmg.DenoisingDataset(data_path=os.path.join(current_file_directory,args.train_data)
                                    ,sigma = args.sigma,patch_size=patch_size,training=True)
    model.train()

    cuda = torch.cuda.is_available()
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    loss_evolution =[]
    min_loss = 1
    print("Début de l'entrainement")
    for epoch in tqdm(range(nb_epoch), desc="Progression des époques",leave=True):

        scheduler.step(epoch)  # step to the learning rate in this epcoh
        
        #Good place ,
        DLoader = DataLoader(dataset=DDataset,batch_size=batch_size, shuffle=True)

        epoch_loss = 0

        for n_count,batch_y, batch_x, noise in enumerate(DLoader):
                
                optimizer.zero_grad()

                if cuda:
                    batch_x, batch_y,noise = batch_x.cuda(), batch_y.cuda(),noise.cuda()

                loss = criterion(model(batch_y), noise)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

        epoch_loss = (epoch_loss/n_count)
        loss_evolution.append(epoch_loss)

        if epoch == 1 : 
             min_loss = epoch_loss
        elif epoch_loss < min_loss:
             min_loss = epoch_loss
             print(f"min_loss achieved for epoch {epoch}")
             torch.save(model, os.path.join(current_file_directory,args.model_dir,args.model_name))
     
    return loss_evolution

loss_evolution = model_training(network=training_network,
                                optimizer=training_optimizer,
                                criterion=training_criterion,
                                scheduler=training_scheduler,
                                nb_epoch=training_nb_epoch,
                                patch_size=training_patch_size,
                                batch_size=training_batch_size)

plt.figure()
plt.semilogy(loss_evolution)
plt.title(f'loss sigma = {args.sigma}')
plt.savefig(os.path.join(current_file_directory,args.loss_dir))
