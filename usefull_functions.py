
import sys
import datetime
import matplotlib.pyplot as plt


def log(*args, **kwargs):
    print('\r',datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)
    sys.stdout.flush()


def results_show(image1,image2,image3,ssim_scores,psnr_scores, figsize=(100,100)):
    """Permet de visualiser l'efficacité du model.
    
    - image 1 : image bruitée
    - image 2 : image débruitée
    - image 3 : vérité terrain"""
    psnr1,psnr2 = psnr_scores
    ssim1,ssim2 = ssim_scores

    f, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=figsize)

    # Affichez chaque image sur son axe respectif
    ax1.imshow(image1)
    ax1.set_title('Image Bruitée'+'\n'+f'PSNR = {psnr1}'+'\n'+f'SSIM = {ssim1}',fontsize = 10)
    ax1.axis('off')  # Désactive les axes pour une meilleure visibilité

    ax2.imshow(image2)
    ax2.set_title('Image Débruitée'+'\n'+f'PSNR = {psnr2}'+'\n'+f'SSIM = {ssim2}',fontsize = 10)
    ax2.axis('off')

    ax3.imshow(image3)
    ax3.set_title('Vérité Terrain',fontsize = 10)
    ax3.axis('off')
    plt.show()