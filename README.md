# Study-of-Low-dose-to-High-dose-CT-using-Supervised-Learning-with-GAN-and-Virtual-Imaging-Trials
This study analyzed the feasibility of utilizing virtually generated CT images to generate high-dose
CT from low-dose CT applying pix2pix cGANs. Different experiments were performed using single
and multiple low-dose to high-dose CT mapping using 4 different pix2pix cGANs. Performance of
all the model are comparable.


### Abstract
Computed tomography (CT) is one of the most widely used radiography exams worldwide for different diagnostic applications. However, CT scans involve ioniz- ing radiational exposure, which raises health concerns. Counter-intuitively, low- ering the adequate CT dose level introduces noise and reduces the image quality, which may impact clinical diagnosis. This study analyzed the feasibility of using a conditional generative adversarial network (cGAN) called pix2pix to learn the mapping from low dose to high dose CT images under different conditions. This study included 270 three-dimensional (3D) CT scan images (85,050 slices) from 90 unique patients imaged virtually using virtual imaging trials platform for model development and testing. Performance was reported as peak signal-to-noise ra- tio (PSNR) and structural similarity index measure (SSIM). Experimental results demonstrated that mapping a single low-dose CT to high-dose CT and weighted two low-dose CTs to high-dose CT have comparable performances using pix2pix CGAN and applicability of using VITs.

### Results
<img src="https://github.com/fitushar/Study-of-Low-dose-to-High-dose-CT-using-Supervised-Learning-with-GAN-and-Virtual-Imaging-Trials/blob/main/figures/visual_results2.png"  width="100%" height="100%">

## Study over-view
* **1.** pix2pix cGAN was adoped for model development.
* **2.** Generating Simulated CT scans utilizing VITs platform.
* **3.** Quantitative and qualitative performance analysis.


### Key refrences:
```ruby
@inproceedings{isola2017image,
  title={Image-to-image translation with conditional adversarial networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1125--1134},
  year={2017}
}

@article{abadi2021virtual,
  title={Virtual imaging trials for coronavirus disease (COVID-19)},
  author={Abadi, Ehsan and Segars, W Paul and Chalian, Hamid and Samei, Ehsan},
  journal={AJR. American journal of roentgenology},
  volume={216},
  number={2},
  pages={362},
  year={2021},
  publisher={NIH Public Access}
}

@inproceedings{tushar2022virtual,
  title={Virtual vs. reality: external validation of COVID-19 classifiers using XCAT phantoms for chest computed tomography},
  author={Tushar, Fakrul Islam and Abadi, Ehsan and Sotoudeh-Paima, Saman and Fricks, Rafael B and Mazurowski, Maciej A and Segars, W Paul and Samei, Ehsan and Lo, Joseph Y},
  booktitle={Medical Imaging 2022: Computer-Aided Diagnosis},
  volume={12033},
  pages={18--24},
  year={2022},
  organization={SPIE}
}
```

## Data and Codes

* **1.** Dataset used in this study is not publicly avaible but can be consider access upon request.
* **1.** All the codes related to this project has been shared publicly within this repo, we will further discussed the code sctructure and in the How to run section.



# How to Run

* **1.** Model were trained using tfrecords and the tfrecords were generated using the coding with the folders

## Genegrate tfrecords for model cGAN-SD, cGAN-Dy codes- https://github.com/fitushar/Study-of-Low-dose-to-High-dose-CT-using-Supervised-Learning-with-GAN-and-Virtual-Imaging-Trials/tree/main/tfrecords_cGAN_SD_and_cGAN_Dy

### Files 
      * (1) Generate_tfrecords-|-> Generate tfrecords for training, testing and validation
```ruby   
            a) config_generate_denoising_tfrecords.py  |-- Configuration tfrecords generation hyperparameter.
            b) Preprocessing_utlities.py               |-- Preprocessing function such as Resampling,Hu cliping, and patch extraction functions                  
            c) main_generate_denoising_tfrecords.py    |-- Generate tfrecords, Main File
            d) CVIT_CT_data.csv                        |-- Data path info
   ```
     * (2) All you need to do is to modify the config_generate_denoising_tfrecords.py for wich data you want generate tfrecors-|-> 
           for cGAN_SD and cGAN_Dy ,USE_THE_DATASET=='CVIT_CT_5p7_28p5_57', and the path of the tfrecords
           
#### config_generate_denoising_tfrecords.py

```ruby
import tensorflow as tf

#-----User input if there were multiple dataset options
USE_THE_DATASET    ='CVIT_CT_5p7_28p5_57'
starting_end_range = False
PATCH_PARAMS = { 'example_size': [256,256], #[hight x width]}
                 'resampling':[2.0,2.0,1.0],# resampling CT
                 'padding_value':-1}

STARTING_INDEX=0
END_INDEX     =1


if USE_THE_DATASET=='CVIT_CT_57_100_200':
    RAW_DATA_DIRECTORY_lower  ='/data/usr/ft42/nobackup/denoising_data_and_tfrecords/RAW_DATA/57mAs_noTCM_nifti_reorient/'
    RAW_DATA_DIRECTORY_medium ='/data/usr/ft42/nobackup/denoising_data_and_tfrecords/RAW_DATA/100mAs_noTCM_nifti_reorient/'
    RAW_DATA_DIRECTORY_high   ='/data/usr/ft42/nobackup/denoising_data_and_tfrecords/RAW_DATA/200mAs_noTCM_nifti_reorient/'

    NAMEOF_TF='DN_57_100_200'

    IMAGE_PATH_INDEX_NAME_lower  ='57mAs'   #index of the column having the data path of nifti.gz (nii.gz)
    IMAGE_PATH_INDEX_NAME_medium ='100mAs'   #index of the column having the data path of nifti.gz (nii.gz)
    IMAGE_PATH_INDEX_NAME_high ='200mAs'   #index of the column having the data path of nifti.gz (nii.gz)

    IMAGE_PATIENT_ID_INDEX_NAME   = 'Patient_ID'
    IMAGE_PHANTOM_ID_INDEX_NAME   = 'Phantom_ID'
    IMAGE_DISEAE_LABEL_INDEX_NAME = 'Diseased_label'


    FLIP_CT='False'
    DATA_CSV='CVIT_CT_data.csv'
    PATH_TO_SAVE_TFRECORDS='/data/usr/ft42/nobackup/denoising_data_and_tfrecords/denoising_tfrecords/tfrecords_DN_57_100_200/DN_57_100_200_tfrecords/'
    SAVING_PNG_OF_THE_PATCH_PNG='/data/usr/ft42/nobackup/denoising_data_and_tfrecords/denoising_tfrecords/tfrecords_DN_57_100_200/DN_57_100_200_tfrecords_png/'
    NAME_OF_PATH_CSV='Denoising_57_100_200_CVIT-Duke_patch1x160x160_spc5x2x2.csv'


if USE_THE_DATASET=='CVIT_CT_5p7_28p5_57':
    RAW_DATA_DIRECTORY_lower  ='/data/usr/ft42/nobackup/denoising_data_and_tfrecords/RAW_DATA/5p7mAs_noTCM_nifti_reorient/'
    RAW_DATA_DIRECTORY_medium ='/data/usr/ft42/nobackup/denoising_data_and_tfrecords/RAW_DATA/28p5mAs_noTCM_nifti_reorient/'
    RAW_DATA_DIRECTORY_high   ='/data/usr/ft42/nobackup/denoising_data_and_tfrecords/RAW_DATA/57mAs_noTCM_nifti_reorient/'

    NAMEOF_TF='DN_5p7_28p5_57'

    IMAGE_PATH_INDEX_NAME_lower  ='5p7mAs'   #index of the column having the data path of nifti.gz (nii.gz)
    IMAGE_PATH_INDEX_NAME_medium ='28p5mAs'   #index of the column having the data path of nifti.gz (nii.gz)
    IMAGE_PATH_INDEX_NAME_high   ='57mAs'   #index of the column having the data path of nifti.gz (nii.gz)

    IMAGE_PATIENT_ID_INDEX_NAME   = 'Patient_ID'
    IMAGE_PHANTOM_ID_INDEX_NAME   = 'Phantom_ID'
    IMAGE_DISEAE_LABEL_INDEX_NAME = 'Diseased_label'


    FLIP_CT='False'
    DATA_CSV='CVIT_CT_data.csv'
    PATH_TO_SAVE_TFRECORDS='/data/usr/ft42/nobackup/denoising_data_and_tfrecords/denoising_tfrecords/tfrecords_DN_5p7_28p5_57/5p7_28p5_57_tfrecords/'
    SAVING_PNG_OF_THE_PATCH_PNG='/data/usr/ft42/nobackup/denoising_data_and_tfrecords/denoising_tfrecords/tfrecords_DN_5p7_28p5_57/5p7_28p5_57_tfrecords_png/'
    NAME_OF_PATH_CSV='Denoising_5p7_28p5_57_CVIT-Duke_patch1x160x160_spc5x2x2.csv'

```


## Similarly to Genegrate tfrecords for model cGAN-DD codes- https://github.com/fitushar/Study-of-Low-dose-to-High-dose-CT-using-Supervised-Learning-with-GAN-and-Virtual-Imaging-Trials/tree/main/tfrecords_cGAN_DD

and use the USE_THE_DATASET=='CVIT_CT_5p7_57' and USE_THE_DATASET=='CVIT_CT_28p5_57'.



## To train, Test and generated predicted images for cGAN-SD, cGAN-DD use codes with in the folder: https://github.com/fitushar/Study-of-Low-dose-to-High-dose-CT-using-Supervised-Learning-with-GAN-and-Virtual-Imaging-Trials/tree/main/Denoising_pix2pix_model

* Configure the model configuration with in the *config_pix2pix_2D_denoising.py* file

```ruby 

import tensorflow as tf
import math
import numpy as np
import pandas as pd
import glob
import os



# The facade training set consist of 400 images


# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 1
# Each image is 192x192x1 in size
IMG_WIDTH  = 256
IMG_HEIGHT = 256

NUMBER_OF_PARALLEL_CALL = 8
PARSHING                = 8
MODEL_TO_RUN ='Model_28p5mAs_to57mAs' # pick which model you want to run 
# for cGAN-SD5.7 MODEL_TO_RUN=='Model_5p7mAs_to57mAs'
# for cGAN-SD28.5 MODEL_TO_RUN=='Model_28p5mAs_to57mAs'
# for cGAN-DD MODEL_TO_RUN=='Model_5P7and28p5mAs_to57mAs'

LAMBDA = 100

ROOT_SAVING_PATHS='/data/usr/ft42/CVIT_Denoising/Denoising_pix2pix_model/'

if MODEL_TO_RUN=='Model_5p7mAs_to57mAs':
    BUFFER_SIZE = 15750
    TRAINING_DATASET_SIZE   = 15750
    VALIDATION_DATASET_SIZE = 6300
    TEST_DATASET_SIZE       = 6300

    LOGDIR             = ROOT_SAVING_PATHS+MODEL_TO_RUN+'/logs/'
    CHECKPOINT_DIR     = ROOT_SAVING_PATHS+MODEL_TO_RUN+'/training_checkpoints'
    GENERATE_IMAGE_DIR = ROOT_SAVING_PATHS+MODEL_TO_RUN+'/generated_denoising_img/'

    TRAIN_TFRECORDS_CSV  = '5p7_28p5_57mAs_train_tfrecords.csv'
    TRAIN_TFRECORDS_PATH = '/data/usr/ft42/nobackup/denoising_data_and_tfrecords/denoising_tfrecords/tfrecords_DN_5p7_28p5_57/5p7_28p5_57_tfrecords/'

    VALIDATION_TFRECORDS_CSV = '5p7_28p5_57mAs_val_tfrecords.csv'
    VALIDATION_TFRECORDS_PATH = '/data/usr/ft42/nobackup/denoising_data_and_tfrecords/denoising_tfrecords/tfrecords_DN_5p7_28p5_57/5p7_28p5_57_tfrecords/'

    TEST_TFRECORDS_CSV  = '5p7_28p5_57mAs_test_tfrecords.csv'
    TEST_TFRECORDS_PATH = '/data/usr/ft42/nobackup/denoising_data_and_tfrecords/denoising_tfrecords/tfrecords_DN_5p7_28p5_57/5p7_28p5_57_tfrecords/'

    DATA_CACHE_TR     = ROOT_SAVING_PATHS+MODEL_TO_RUN+"_train"
    DATA_CACHE_VAL    = ROOT_SAVING_PATHS+MODEL_TO_RUN+"_val"
    VALIDATION_PSNR_SSIM_CSV_PATH = ROOT_SAVING_PATHS+MODEL_TO_RUN+'/'+MODEL_TO_RUN+'_validation_df.csv'
    GENERATE_IMAGE_CSV= '5p7_28p5_57mAs_Generate_tfrecords.csv'

if MODEL_TO_RUN=='Model_28p5mAs_to57mAs':

    BUFFER_SIZE = 15750
    TRAINING_DATASET_SIZE   = 15750
    VALIDATION_DATASET_SIZE = 6300
    TEST_DATASET_SIZE       = 6300

    LOGDIR             = ROOT_SAVING_PATHS+MODEL_TO_RUN+'/logs/'
    CHECKPOINT_DIR     = ROOT_SAVING_PATHS+MODEL_TO_RUN+'/training_checkpoints'
    GENERATE_IMAGE_DIR = ROOT_SAVING_PATHS+MODEL_TO_RUN+'/generated_denoising_img/'

    TRAIN_TFRECORDS_CSV  = '5p7_28p5_57mAs_train_tfrecords.csv'
    TRAIN_TFRECORDS_PATH = '/data/usr/ft42/nobackup/denoising_data_and_tfrecords/denoising_tfrecords/tfrecords_DN_5p7_28p5_57/5p7_28p5_57_tfrecords/'

    VALIDATION_TFRECORDS_CSV =  '5p7_28p5_57mAs_val_tfrecords.csv'
    VALIDATION_TFRECORDS_PATH = '/data/usr/ft42/nobackup/denoising_data_and_tfrecords/denoising_tfrecords/tfrecords_DN_5p7_28p5_57/5p7_28p5_57_tfrecords/'

    TEST_TFRECORDS_CSV  = '5p7_28p5_57mAs_test_tfrecords.csv'
    TEST_TFRECORDS_PATH = '/data/usr/ft42/nobackup/denoising_data_and_tfrecords/denoising_tfrecords/tfrecords_DN_5p7_28p5_57/5p7_28p5_57_tfrecords/'

    DATA_CACHE_TR     = ROOT_SAVING_PATHS+MODEL_TO_RUN+"_train"
    DATA_CACHE_VAL    = ROOT_SAVING_PATHS+MODEL_TO_RUN+"_val"
    VALIDATION_PSNR_SSIM_CSV_PATH = ROOT_SAVING_PATHS+MODEL_TO_RUN+'/'+MODEL_TO_RUN+'_validation_df.csv'
    GENERATE_IMAGE_CSV= '5p7_28p5_57mAs_Generate_tfrecords.csv'


if MODEL_TO_RUN=='Model_5P7and28p5mAs_to57mAs':

    BUFFER_SIZE = 15750*2
    TRAINING_DATASET_SIZE   = 15750*2
    VALIDATION_DATASET_SIZE = 6300*2
    TEST_DATASET_SIZE       = 6300*2

    LOGDIR             = ROOT_SAVING_PATHS+MODEL_TO_RUN+'/logs/'
    CHECKPOINT_DIR     = ROOT_SAVING_PATHS+MODEL_TO_RUN+'/training_checkpoints'
    GENERATE_IMAGE_DIR = ROOT_SAVING_PATHS+MODEL_TO_RUN+'/generated_denoising_img/'

    TRAIN_TFRECORDS_CSV  = '5p7And28p5_57mAs_train_tfrecords.csv'
    TRAIN_TFRECORDS_PATH = '/data/usr/ft42/nobackup/denoising_data_and_tfrecords/denoising_tfrecords/tfrecords_DN_5p7and28p5_57/5p7and28p5_57_tfrecords/'

    VALIDATION_TFRECORDS_CSV =  '5p7And28p5_57mAs_val_tfrecords.csv'
    VALIDATION_TFRECORDS_PATH = '/data/usr/ft42/nobackup/denoising_data_and_tfrecords/denoising_tfrecords/tfrecords_DN_5p7and28p5_57/5p7and28p5_57_tfrecords/'

    TEST_TFRECORDS_CSV  = '5p7And28p5_57mAs_test_tfrecords.csv'
    TEST_TFRECORDS_PATH = '/data/usr/ft42/nobackup/denoising_data_and_tfrecords/denoising_tfrecords/tfrecords_DN_5p7and28p5_57/5p7and28p5_57_tfrecords/'
     
    GENERATE_IMAGE_CSV  = '5p7And28p5_57mAs_Generate_tfrecords.csv'

    DATA_CACHE_TR     = ROOT_SAVING_PATHS+MODEL_TO_RUN+"_train"
    DATA_CACHE_VAL    = ROOT_SAVING_PATHS+MODEL_TO_RUN+"_val"
    VALIDATION_PSNR_SSIM_CSV_PATH = ROOT_SAVING_PATHS+MODEL_TO_RUN+'/'+MODEL_TO_RUN+'_validation_df.csv'

```


* to train: python train_pix2pix_2D_denoising.py 
* to test : python test_pix2pix_2D_denoising.py
* to generate predicted denoised image : python generate_image_numpy.py 



## similarly to train, Test and generated predicted images for cGAN-Dy use codes with in the folder: https://github.com/fitushar/Study-of-Low-dose-to-High-dose-CT-using-Supervised-Learning-with-GAN-and-Virtual-Imaging-Trials/tree/main/cGAN_Dy
