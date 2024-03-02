import data_models_generator as dmg
import torch.optim as optim
import torch,os
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import usefull_functions as uf

current_file_directory = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description='PyTorch DnCNN')
parser.add_argument('--sigma', default=25, type=int, help="Choix du sigma à utliser pour ,l'entrainement et le test. Attention un sigma de -1 rendra le sigma alétoire entre 0 et 50")
parser.add_argument('--image_channels', default=3,type=int, help="permet de choisir si les images traitée sont en rgb ou en niveau de gris")

##Données d'entrainement
parser.add_argument('--train_data', default=os.path.join(current_file_directory,
                                                         'Data_rgb/Train'), type=str, help='path of train data')
#Paramètre de sauvegarde
parser.add_argument('--model_dir', default=os.path.join(current_file_directory,
                                                        'Models/models_rgb/sigma_25'), help='directory of the model')
parser.add_argument('--loss_dir', default=os.path.join(current_file_directory,
                                                       'losses/loss'), help='directory of the loss plot')
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
args, unknown = parser.parse_known_args()


###Choix des hyper_paramètres
training_batch_size = 1
training_nb_epoch = 1005
training_learning_rate = 1e-3
training_patch_size = 35

training_network = dmg.DnCNN(image_channels=args.image_channels)
training_optimizer = optim.Adam(training_network.parameters(), lr=training_learning_rate)
training_criterion = F.mse_loss
training_scheduler = MultiStepLR(training_optimizer, milestones=[training_nb_epoch/2], gamma=0.1) 


def model_training(network,optimizer,criterion,scheduler,nb_epoch,patch_size,batch_size) :
    """Entraînement et enregistement du modèle avec les options mises en paramètre
    
    network : type de réseau à entraîner
    optimizer : optimseur choisi pour l'optimisation
    criterion : critère choisi pour l'optimisation
    scheduler : opérateur de changement de learning rate choisi pour l'entraînement
    nb_epoch : nombre d'epoch maximal de l'entraînement
    patch_size : taille des patch des images d'entraînement
    batch_size : taille des batch utilisée pour l'optimisation"""
    print('===> Building model')
    

    model = network
    initial_epoch,model_name = uf.find_last_checkpoint(save_dir=args.model_dir) 
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        model = torch.load(model_name)
    else :
        print('Starting training form epoch 0')
    model.train()

   
    DDataset = dmg.DenoisingDataset(data_path=args.train_data,sigma = args.sigma,
                                    patch_size=patch_size,training=True)
    

    cuda = torch.cuda.is_available()
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    loss_evolution =[]
    min_epoch = initial_epoch
    print("Training start")
    for start,epoch in enumerate(tqdm(range(initial_epoch,nb_epoch), desc="epoch progression",leave=True)):
        
        DLoader = DataLoader(dataset=DDataset,batch_size=batch_size, shuffle=True)
        epoch_loss = 0

        for n_count,(batch_y, batch_x, noise) in enumerate(DLoader):
                
                if cuda:
                    batch_x, batch_y,noise = batch_x.cuda(), batch_y.cuda(),noise.cuda()

                optimizer.zero_grad()
                loss = criterion(model(batch_y), noise)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
        
        scheduler.step()

        epoch_loss = (epoch_loss/n_count)
        loss_evolution.append(epoch_loss)

        if start == 0 : 
             min_loss = epoch_loss
        elif epoch_loss < min_loss:
             min_epoch = epoch
             min_loss = epoch_loss
             torch.save(model,os.path.join(args.model_dir,f'model_c{args.image_channels}_s{args.sigma}_e{epoch}.pth'))

    print(f"min_loss achieved for epoch {min_epoch}")
    return loss_evolution

loss_evolution = model_training(network=training_network,
                                optimizer=training_optimizer,
                                criterion=training_criterion,
                                scheduler=training_scheduler,
                                nb_epoch=training_nb_epoch,
                                patch_size=training_patch_size,
                                batch_size=training_batch_size)

plt.figure()
plt.semilogy(range(training_nb_epoch-len(loss_evolution),training_nb_epoch),loss_evolution)
plt.title(f'loss sigma = {args.sigma}')
plt.savefig(args.loss_dir)
