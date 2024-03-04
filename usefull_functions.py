
import sys,glob,os,datetime
import matplotlib.pyplot as plt
import numpy as np

def log(*args, **kwargs):
    print('\r',datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)
    sys.stdout.flush()


def results_show(image1,image2,image3,ssim_scores,psnr_scores, figsize=(100,100)):
    """Permet de visualiser l'efficacité du modèle.
    
    - image 1 : image bruitée
    - image 2 : image débruitée
    - image 3 : vérité terrain"""
    psnr1,psnr2 = psnr_scores
    ssim1,ssim2 = ssim_scores

    f, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=figsize)

    ax1.imshow(image1)
    ax1.set_title('Image Bruitée'+'\n'+f'PSNR = {psnr1}'+'\n'+f'SSIM = {ssim1}',fontsize = 10)
    ax1.axis('off') 

    ax2.imshow(image2)
    ax2.set_title('Image Débruitée'+'\n'+f'PSNR = {psnr2}'+'\n'+f'SSIM = {ssim2}',fontsize = 10)
    ax2.axis('off')

    ax3.imshow(image3)
    ax3.set_title('Vérité Terrain',fontsize = 10)
    ax3.axis('off')
    plt.show()


def batch_visualisation(batch_in):
    """permet d'extraire les images liées au batch en entrée pour pouvoir calculer des scores ou l'afficher
    
    - batch_in : batch de taille 1 à traiter
    - batch_out : image numpy convertie"""
    batch_out = batch_in.squeeze(0).permute(1, 2, 0)
    batch_out = batch_out.cpu()
    batch_out = batch_out.detach().numpy().astype(np.float32)

    return batch_out


def find_last_checkpoint(save_dir):
    """Trouve le model associé à l'entrainement proposé. Donne le denier epoch utilisé pour l'entrainer.
    
    - save_dir : chemin de sauvegarde du model.
    - initial_epoch : epoch à partir du quel commencer l'entrainement.
    - model_name : nom du model associé à l'entrainement choisi."""
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    initial_epoch = 0
    model_name = ''
    if file_list:

        for file_ in file_list:
            result = file_.split('_')[-1]
            result = result.replace('.pth','')
            result = result.replace('e','')
            try : 
                int(result)
            except :
                print("Models with wrong names are detected")
            else :  
                if int(result) > initial_epoch :
                    initial_epoch = int(result)
                    model_name = file_
                  
    return initial_epoch,model_name