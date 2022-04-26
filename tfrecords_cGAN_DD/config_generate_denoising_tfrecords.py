import tensorflow as tf

#-----User input if there were multiple dataset options
USE_THE_DATASET    ='CVIT_CT_28p5_57'
starting_end_range = False
PATCH_PARAMS = { 'example_size': [256,256], #[hight x width]}
                 'resampling':[2.0,2.0,1.0],# resampling CT
                 'padding_value':-1}




if USE_THE_DATASET=='CVIT_CT_5p7_57':
    RAW_DATA_DIRECTORY_lower  ='/data/usr/ft42/nobackup/denoising_data_and_tfrecords/RAW_DATA/5p7mAs_noTCM_nifti_reorient/'
    RAW_DATA_DIRECTORY_high   ='/data/usr/ft42/nobackup/denoising_data_and_tfrecords/RAW_DATA/57mAs_noTCM_nifti_reorient/'

    NAMEOF_TF='DN_5p7_57'

    IMAGE_PATH_INDEX_NAME_lower  ='5p7mAs'   #index of the column having the data path of nifti.gz (nii.gz)
    IMAGE_PATH_INDEX_NAME_high   ='57mAs'   #index of the column having the data path of nifti.gz (nii.gz)

    IMAGE_PATIENT_ID_INDEX_NAME   = 'Patient_ID'
    IMAGE_PHANTOM_ID_INDEX_NAME   = 'Phantom_ID'
    IMAGE_DISEAE_LABEL_INDEX_NAME = 'Diseased_label'


    FLIP_CT='False'
    DATA_CSV='CVIT_CT_data.csv'
    PATH_TO_SAVE_TFRECORDS='/data/usr/ft42/nobackup/denoising_data_and_tfrecords/denoising_tfrecords/tfrecords_DN_5p7_57/5p7_57_tfrecords/'
    SAVING_PNG_OF_THE_PATCH_PNG='/data/usr/ft42/nobackup/denoising_data_and_tfrecords/denoising_tfrecords/tfrecords_DN_5p7_57/5p7_57_tfrecords_png/'
    NAME_OF_PATH_CSV='Denoising_5p7_57_CVIT-Duke_patch1x160x160_spc5x2x2.csv'

if USE_THE_DATASET=='CVIT_CT_28p5_57':
    RAW_DATA_DIRECTORY_lower  ='/data/usr/ft42/nobackup/denoising_data_and_tfrecords/RAW_DATA/28p5mAs_noTCM_nifti_reorient/'
    RAW_DATA_DIRECTORY_high   ='/data/usr/ft42/nobackup/denoising_data_and_tfrecords/RAW_DATA/57mAs_noTCM_nifti_reorient/'

    NAMEOF_TF='DN_28p5_57'

    IMAGE_PATH_INDEX_NAME_lower  ='28p5mAs'   #index of the column having the data path of nifti.gz (nii.gz)
    IMAGE_PATH_INDEX_NAME_high   ='57mAs'   #index of the column having the data path of nifti.gz (nii.gz)

    IMAGE_PATIENT_ID_INDEX_NAME   = 'Patient_ID'
    IMAGE_PHANTOM_ID_INDEX_NAME   = 'Phantom_ID'
    IMAGE_DISEAE_LABEL_INDEX_NAME = 'Diseased_label'


    FLIP_CT='False'
    DATA_CSV='CVIT_CT_data.csv'
    PATH_TO_SAVE_TFRECORDS='/data/usr/ft42/nobackup/denoising_data_and_tfrecords/denoising_tfrecords/tfrecords_DN_28p5_57/28p5_57_tfrecords/'
    SAVING_PNG_OF_THE_PATCH_PNG='/data/usr/ft42/nobackup/denoising_data_and_tfrecords/denoising_tfrecords/tfrecords_DN_28p5_57/28p5_57_tfrecords_png/'
    NAME_OF_PATH_CSV='Denoising_28p5_57_CVIT-Duke_patch1x160x160_spc5x2x2.csv'
