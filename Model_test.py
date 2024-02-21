from IPython.display import display, clear_output
import torch
import usefull_functions as uf
import data_models_generator as dmg
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import argparse
from skimage.metrics import peak_signal_noise_ratio as psnr
#from skimage.metrics import structural_similarity as ssim
import time
import json
from tqdm import tqdm
import usefull_functions as uf


parser = argparse.ArgumentParser(description='PyTorch DnCNN')
parser.add_argument('--model_name', default='model_rgb_1.pth', type=str, help="Nom du model qu'on souhaite entrainer ou tester")
parser.add_argument('--sigma', default=25, type=int, help="Choix du sigma à utliser pour ,l'entrainement et le test. Attention un sigma de -1 rendra le sigma alétoire entre 0 et 50")
parser.add_argument('--model_dir', default='/Models/', help='directory of the model')
parser.add_argument('--test_data', default='/Projet/Data_rgb/Train', type=str, help='path of test data')
parser.add_argument('--result_file',default='Scores PSNR débruitage.json',type = str,help = 'name of the file containing the results of the test')
parser.add_argument('--visualisation',default=0,type=int,help="Choisi d'afficher les images débruitée ou non")
args, unknown = parser.parse_known_args()

current_file_directory = os.path.dirname(os.path.abspath(__file__))
psnr_results = {
    'whatin': 'Contient les listes des psnr obtenus pour chaque sigma',
}

def model_test() :

    psnr_noise_images = []
    psnr_denoise_images = []
    denoising_time = []

    cuda = torch.cuda.is_available()
    if cuda :
        model = torch.load(os.path.join(current_file_directory,args.model_dir,args.model_name)).cuda()
    else :
        model = torch.load(os.path.join(current_file_directory,args.model_dir,args.model_name))

    uf.log('load trained model')

    model.eval()
    DDataset = dmg.DenoisingDataset(data_path=os.path.join(current_file_directory,args.test_data)
                                    ,sigma = args.sigma,training=False)
    DLoader = DataLoader(dataset=DDataset,batch_size=1, shuffle=True)

    with torch.no_grad():
        for batch_y, batch_x, noise in tqdm(DLoader,f" Chargement des images sig = {args.sig}"):

            if cuda :
                torch.cuda.synchronize()
                batch_x, batch_y,noise = batch_x.cuda(), batch_y.cuda(), noise.cuda()
                
            start_time = time.time()
            batch_predicted = model(batch_y)
            elapsed_time = time.time() - start_time
            denoising_time.append(elapsed_time)

            batch_predicted = batch_y - batch_predicted

            batch_x = batch_x.squeeze(0).permute(1, 2, 0)
            batch_x = batch_x.cpu()
            batch_x = batch_x.detach().numpy().astype(np.float32)

            batch_y = batch_y.squeeze(0).permute(1, 2, 0)
            batch_y = batch_y.cpu()
            batch_y = batch_y.detach().numpy().astype(np.float32)

            batch_predicted = batch_predicted.squeeze(0).permute(1, 2, 0)
            batch_predicted = batch_predicted.cpu()
            batch_predicted = batch_predicted.detach().numpy().astype(np.float32)

            psnr_noise = psnr(batch_x,batch_y,data_range = 1.0)
            psnr_denoise = psnr(batch_x,batch_predicted,data_range = 1.0)

            psnr_noise_images.append(psnr_noise)
            psnr_denoise_images.append(psnr_denoise)

            if args.visualisation :
                uf.results_show(image1=batch_x,image2=batch_predicted,image3=batch_y,
                             psnr_scores=(psnr_noise,psnr_denoise),
                             ssim_scores=(1,1))
               
    name_d = str(args.sigma) + '_d'
    name_b = str(args.sigma) + '_b'
    name_t = str(args.sigma) + '_t'
    psnr_results[name_b] = psnr_noise_images
    psnr_results[name_d] = psnr_denoise_images
    psnr_results[name_t] = denoising_time


    with open(args.result_file, 'w') as fichier:
        json.dump(psnr_results, fichier)

    return 

model_test()