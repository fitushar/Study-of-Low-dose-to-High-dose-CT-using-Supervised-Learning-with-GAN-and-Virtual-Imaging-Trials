
#--- Import libraries--------
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import pandas as pd
import SimpleITK as sitk
import sys
import cv2
from config_generate_denoising_tfrecords import*
import math
import random
from scipy import ndimage
from skimage.filters import threshold_mean
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from Preprocessing_utlities import resample_img2mm
from Preprocessing_utlities import normalise_one_one
from Preprocessing_utlities import resize_image_with_crop_or_pad
from matplotlib import pyplot as plt
import pathlib
random.seed(3)

########################-------Fucntions for tf records-----###########################

# The following functions can be used to convert a value to a type compatible with tf.Example.
# this part of the code is copped from tensorflow official docuumentation Totorial,
#ref: https://www.tensorflow.org/tutorials/load_data/tfrecord

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def flow_from_df(dataframe: pd.DataFrame, chunk_size):
    for start_row in range(0, dataframe.shape[0], chunk_size):
        end_row  = min(start_row + chunk_size, dataframe.shape[0])
        yield dataframe.iloc[start_row:end_row, :]
########################---------------------------------###########################
#plot function

def Save_png(a_id,img1,img2,img3,path):


    p_id=a_id
    print(p_id)

    f, axarr = plt.subplots(1,3,figsize=(12,4));
    #f.suptitle('Id-->{}'.format(p_id),fontsize =12)

    ########--------------------------Row---1---------###############
    img_plot = axarr[0].imshow(np.squeeze(img1), cmap='gray',origin='lower');
    axarr[0].axis('off')
    #axarr[0].set_title('57mAs',fontsize =12)

    middle_plot = axarr[1].imshow(np.squeeze(img2), cmap='gray',origin='lower');
    axarr[1].axis('off')
    #axarr[1].set_title('100mAs',fontsize =12)

    last_plot = axarr[2].imshow(np.squeeze(img3), cmap='gray',origin='lower');
    axarr[2].axis('off')
    #axarr[2].set_title('200 mAs',fontsize = 12)


    plt.tight_layout()
    png_name=path
    plt.savefig(png_name)
    plt.close()
    return

###################################################################################
def generate_denoising_tfrecords():

    #==| Raed CSV
    read_csv=pd.read_csv(DATA_CSV,keep_default_na=False,na_values=[])
    #==| Getting parameters file
    patch_params=PATCH_PARAMS
    #----Listing the cases-----
    lower_dose_ct_name_list  = read_csv[IMAGE_PATH_INDEX_NAME_lower].tolist()
    
    high_dose_ct_name_list   = read_csv[IMAGE_PATH_INDEX_NAME_high].tolist()

    patient_name_list = read_csv[IMAGE_PATIENT_ID_INDEX_NAME].tolist()
    phantom_name_list = read_csv[IMAGE_PHANTOM_ID_INDEX_NAME].tolist()
    disease_label_list= read_csv[IMAGE_DISEAE_LABEL_INDEX_NAME].tolist()


    print('Total Dataset size = {}'.format(len(read_csv)))

    #----------Creating list to store the data

    crt_phantom_id_list    = []
    crt_ct_id_list         = []
    crt_slide_number_list  = []
    crt_tfrecords_name     = []
    crt_patint_covid_label = []

    if starting_end_range:
        df_start=STARTING_INDEX
        df_end  =END_INDEX
    else:
        df_start=0
        df_end  =len(read_csv)

    pathlib.Path(PATH_TO_SAVE_TFRECORDS).mkdir(parents=True, exist_ok=True)
    pathlib.Path(SAVING_PNG_OF_THE_PATCH_PNG).mkdir(parents=True, exist_ok=True)


    for ct_data_i in range(df_start,df_end):

        #-----CT path
        img_path_lower     = RAW_DATA_DIRECTORY_lower  + lower_dose_ct_name_list[ct_data_i]
        img_path_high      = RAW_DATA_DIRECTORY_high   + high_dose_ct_name_list[ct_data_i]

        diseased_lbl       = disease_label_list[ct_data_i]
        patient_id         = patient_name_list[ct_data_i]
        phantom_id         = phantom_name_list[ct_data_i]

        print('Phantom-ID:{},Patient-ID:{},Label={}'.format(phantom_id,patient_id,diseased_lbl))


        #--|CT-lower
        img_sitk_lower = sitk.ReadImage(img_path_lower, sitk.sitkFloat32) #ReadImage
        img_sitk_lower = resample_img2mm(img_sitk_lower, out_spacing=[2.0, 2.0, 1.0], is_label=False) #resampling
        image_lower    = sitk.GetArrayFromImage(img_sitk_lower) #get numpy image from array
        if FLIP_CT=='True':
           image_lower= image_lower[:, ::-1, :]
        image_lower    = np.clip(image_lower, -1000., 500.).astype(np.float32)
        image_lower    = normalise_one_one(image_lower)
        print('CT-lower Preprocessing done !')


        #--|CT-high
        img_sitk_high = sitk.ReadImage(img_path_high, sitk.sitkFloat32)
        img_sitk_high = resample_img2mm(img_sitk_high, out_spacing=[2.0, 2.0, 1.0], is_label=False)
        image_high    = sitk.GetArrayFromImage(img_sitk_high)
        if FLIP_CT=='True':
           image_high= image_high[:, ::-1, :]
        image_high    = np.clip(image_high, -1000., 500.).astype(np.float32)
        image_high    = normalise_one_one(image_high)
        print('CT-high Preprocessing done !')


        print('CT-lower-Shape---{}'.format(image_lower.shape))
        print('CT-high-Shape---{}'.format(image_high.shape))



        #patch_name =bytes(subject_id, 'utf-8')

        for ct_slice_j in range(30,345):

            print('processing_slice-{}'.format(ct_slice_j))

            #--Getting the sllices
            slice_lower  =  image_lower[ct_slice_j,:,:]
            slice_high   =  image_high[ct_slice_j,:,:]

            #--crop/padding to exact size
            slice_lower  = resize_image_with_crop_or_pad(slice_lower, patch_params['example_size'], mode='constant',constant_values=patch_params['padding_value'])
            slice_high   = resize_image_with_crop_or_pad(slice_high, patch_params['example_size'], mode='constant',constant_values=patch_params['padding_value'])

            slice_number      = ct_slice_j
            tfrecords_name    = NAMEOF_TF+'_{}_l_{}_s_{}.tfrecords'.format(patient_id,diseased_lbl,slice_number)
            slice_tfrecord_id = NAMEOF_TF+'_{}_l_{}_s_{}'.format(patient_id,diseased_lbl,slice_number)
            slice_tfrecord_id = bytes(slice_tfrecord_id, 'utf-8')
            record_mask_file =PATH_TO_SAVE_TFRECORDS+tfrecords_name

            slice_png_path = SAVING_PNG_OF_THE_PATCH_PNG+NAMEOF_TF+'_{}_l_{}_s_{}.png'.format(patient_id,diseased_lbl,slice_number)
            splice_id      = NAMEOF_TF+'_{}_l_{}_s_{}'.format(patient_id,diseased_lbl,slice_number)

            crt_phantom_id_list.append(phantom_id)
            crt_ct_id_list.append(patient_id)
            crt_slide_number_list.append(slice_number)
            crt_tfrecords_name.append(tfrecords_name)
            crt_patint_covid_label.append(diseased_lbl)
            with tf.io.TFRecordWriter(record_mask_file) as writer:
                feature = {
                               'covid_lbl': _int64_feature(diseased_lbl),
                                'lower_img':_bytes_feature(slice_lower.tostring()),
                                'high_img':_bytes_feature(slice_high.tostring()),
                                'Patch_h':_int64_feature(patch_params['example_size'][0]), #h
                                'Patch_w':_int64_feature(patch_params['example_size'][1]), #w
                                'Sub_id':_bytes_feature(slice_tfrecord_id)
                                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
            writer.close()


            #Save_png(a_id=splice_id,img1=slice_lower,img2=slice_medium,img3=slice_high,path=slice_png_path)

    tf_info = pd.DataFrame(list(zip(crt_phantom_id_list,crt_ct_id_list,crt_slide_number_list,crt_patint_covid_label,crt_tfrecords_name)),columns=['Phantom_ID','Patient_ID','Slice_num','diseased_lbl','tf_records_name'])
    tf_info.to_csv(NAME_OF_PATH_CSV, encoding='utf-8', index=False)

    return


if __name__ == '__main__':
    generate_denoising_tfrecords()
