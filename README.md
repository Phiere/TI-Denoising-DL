# TI-Denoising-DL
Work inspired by the articles:

1 - Learning Deep CNN Denoiser Prior for Image Restoration
by Kai Zhang1,2, Wangmeng Zuo1, Shuhang Gu2, Lei Zhang2
1School of Computer Science and Technology, Harbin Institute of Technology, Harbin, China 2Dept. of Computing, The Hong Kong Polytechnic University, Hong Kong, China

2 - Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising by
Kai Zhang, Wangmeng Zuo, Senior Member, IEEE, Yunjin Chen, Deyu Meng, and Lei Zhang, Senior Member, IEEE

The purpose of this work is to implement a deep-learning denoising method for Gaussian noise.

## Database used:

To train our results, we use the BSD68 database comprising 68 images.

![BSD68 Excerpts](read_me_images/image1.png)

## Methods used:

-  We start by augmenting our database using the following transformations:

    - randomCrop 35x35 (network use square patches)
    - randomFlip (horizontal,vertical)
    - randomRotation(30 deg)
    - normalisation 


- The network we use is the following:

![Network representation diagram](read_me_images/image2.png)

### Parameters:
- optimizer : adam
- criterion : F.mse_loss
- learning rate : 1e-3 then 1e-4

## Results:

For denoising with a sigma of 25, we obtain the following results:

![Denoising result sigma = 25](read_me_images/image3.png)

For other sigmas, the results obtained are as follows:

![Denoising result BSD68](read_me_images/image4.png)

## Implementing for another (flowers) database:

![Denoising result other database (flowers)](read_me_images/image5.png)

## Use for personnal photos

To use the trained network, you will need to provide the path of the photo to be denoised in the photo_application function of Model_test.py. As the noise sigma of the photo is not known, it may be necessary to train a model for the studied sigma. Just run the Model_train.py script choosing the right sigma.

##Conclusion:

We have successfully managed to create and use a deep learning model to denoise images to which Gaussian noise had been added. Our model still struggles to generalize to other databases.