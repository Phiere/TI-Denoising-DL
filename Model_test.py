import torch, os, argparse, time
import usefull_functions as uf
import data_models_generator as dmg
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


current_file_directory = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description='PyTorch DnCNN')
parser.add_argument('--model_name', default='model_rgb_1.pth', type=str, help="Nom du model qu'on souhaite entrainer ou tester")
parser.add_argument('--sigma', default=25, type=int, help="Choix du sigma à utliser pour ,l'entrainement et le test. Attention un sigma de -1 rendra le sigma alétoire entre 0 et 50")
parser.add_argument('--model_dir', default=os.path.join(current_file_directory,'Models/')
                    , type = str, help='directory of the model')
parser.add_argument('--test_data', default=os.path.join(current_file_directory,'/Projet/Data_rgb/Train')
                    , type=str, help='path of test data')
parser.add_argument('--visualisation',default=0,type=int,help="Choisi d'afficher les images débruitée ou non")
args, unknown = parser.parse_known_args()


def model_test() :
    """Fonction de test sur la base de données en paramètre du réseau utilisé. Il est possible de visualiser les résultats obtenus
    avec le paramètre visualisation
    
    psnr_noise_images,psnr_denoise_images : scores psnr des images bruitées et débruitées
    ssim_noise_images,ssim_denoise_images : scores ssim des images bruitées et débruitées
    denoising_time : temps de calcul pour le modèle utilisé
    """
    psnr_noise_images = []
    psnr_denoise_images = []
    ssim_noise_images = []
    ssim_denoise_images = []
    denoising_time = []

    cuda = torch.cuda.is_available()
    model = torch.load(os.path.join(current_file_directory,args.model_dir,args.model_name))

    if cuda :
        torch.cuda.synchronize()
        model.cuda()
        uf.log('load trained model on gpu')
    else :
        uf.log('load trained model on cpu')

    model.eval()
    DDataset = dmg.DenoisingDataset(data_path=os.path.join(current_file_directory,args.test_data)
                                    ,sigma = args.sigma)
    DLoader = DataLoader(dataset=DDataset,batch_size=1, shuffle=True)

    with torch.no_grad():
        for batch_y, batch_x,_ in tqdm(DLoader,f" Chargement des images sig = {args.sig}"):

            if cuda :
                batch_x, batch_y= batch_x.cuda(), batch_y.cuda()
                
            #Denoising
            start_time = time.time()
            noise_predicted = model(batch_y)
            elapsed_time = time.time() - start_time
            denoising_time.append(elapsed_time)
            batch_predicted = batch_y - noise_predicted

            #Results transformations for calculs
            batch_x = uf.batch_visualisation(batch_x)
            batch_y = uf.batch_visualisation(batch_y)
            batch_predicted = uf.batch_visualisation(batch_predicted)
            
            #PSNR calculs 
            psnr_noise = psnr(batch_x,batch_y,data_range = 1.0)
            psnr_denoise = psnr(batch_x,batch_predicted,data_range = 1.0)
            psnr_noise_images.append(psnr_noise)
            psnr_denoise_images.append(psnr_denoise)
            
            #SSIM calculs 
            ssim_index_noise,_ = ssim(batch_x,batch_y,channel_axis = -1,full=True)
            ssim_index_denoise,_ = ssim(batch_x,batch_predicted,channel_axis = -1,full=True)
            ssim_noise_images.append(ssim_index_noise)
            ssim_denoise_images.append(ssim_index_denoise)

            if args.visualisation :
                uf.results_show(image1=batch_x,image2=batch_predicted,image3=batch_y,
                             psnr_scores=(psnr_noise,psnr_denoise),
                             ssim_scores=(ssim_index_noise,ssim_index_denoise))
               
    return  psnr_noise_images,psnr_denoise_images,ssim_noise_images,ssim_denoise_images,denoising_time


def photo_application(photo_path,model_path):
    """Mise en application du modèle pour des images déjà bruitées
    
    photo_path : chemin de la photo à débruitée
    model_path : chemin du modèle utilisé pour le débruitage"""

    model = torch.load(os.path.join(current_file_directory,model_path))

    model.eval()
    DDataset = dmg.PersonalDenoiserFataset(data_path=os.path.join(current_file_directory,photo_path)
                                    ,sigma = args.sigma)
    DLoader = DataLoader(dataset=DDataset,batch_size=1, shuffle=True)

    with torch.no_grad():
        for  batch_x,_ in tqdm(DLoader,f" Chargement des images sig = {args.sig}"):
                
            #Denoising
            noise_predicted = model(batch_x)
            batch_predicted = batch_x - noise_predicted

            #Results transformations for calculs
            batch_x = uf.batch_visualisation(batch_x)
            batch_predicted = uf.batch_visualisation(batch_predicted)
            
            #Display Results
            f, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(batch_x)
            ax1.set_title('Image de base')
            ax1.axis('off') 
            ax2.imshow(batch_predicted)
            ax2.set_title('Image Débruitée')
            ax2.axis('off')
               
    return  

