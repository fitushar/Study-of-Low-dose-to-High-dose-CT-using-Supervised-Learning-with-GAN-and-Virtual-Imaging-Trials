import tensorflow as tf
import math
import numpy as np
import pandas as pd
import glob
import os



# The facade training set consist of 400 images
BUFFER_SIZE = 15750
TRAINING_DATASET_SIZE   = 15750
VALIDATION_DATASET_SIZE = 6300
TEST_DATASET_SIZE       = 6300

# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 1
# Each image is 192x192x1 in size
IMG_WIDTH  = 192
IMG_HEIGHT = 192

NUMBER_OF_PARALLEL_CALL = 8
PARSHING                = 8
MODEL_TO_RUN ='Model_dual'
LAMBDA = 100

ROOT_SAVING_PATHS='/data/usr/ft42/CVIT_Denoising/Denoising_pix2pix_DualDatamodel/'

if MODEL_TO_RUN=='Model_dual':

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
    GENERATE_IMAGE_CSV = '5p7_28p5_57mAs_Generate_tfrecords.csv'
