from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np

from volumentations import *


########################-------Fucntions for tf records
# The following functions can be used to convert a value to a type compatible
# with tf.Example.
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


@tf.function
def decode_lower_ct(Serialized_example):
    features = {
                'covid_lbl': tf.io.FixedLenFeature([],tf.int64),
                'lower_img': tf.io.FixedLenFeature([],tf.string),
                'medium_img':tf.io.FixedLenFeature([],tf.string),
                'high_img':tf.io.FixedLenFeature([],tf.string),
                'Patch_h': tf.io.FixedLenFeature([],tf.int64), #h
                'Patch_w': tf.io.FixedLenFeature([],tf.int64), #w
                'Sub_id': tf.io.FixedLenFeature([],tf.string)
                                }

    examples    = tf.io.parse_single_example(Serialized_example,features)
    img_shape   = [256,256]
    ##Decode_image_float
    image_1     = tf.io.decode_raw(examples['lower_img'], float) ##Decode_image_float
    input_image = tf.reshape(image_1,img_shape)        #Resgapping_the_data
    input_image = tf.expand_dims(input_image, axis=-1) #Because CNN expect(batch,H,W,D,CHANNEL)
    input_image = tf.cast(input_image, tf.float32)     #Because CNN expect(batch,H,W,D,CHANNEL)


    image_2    = tf.io.decode_raw(examples['high_img'], float) ##Decode_image_float
    real_image = tf.reshape(image_2,img_shape)        #Resgapping_the_data
    real_image = tf.expand_dims(real_image, axis=-1) #Because CNN expect(batch,H,W,D,CHANNEL)
    real_image = tf.cast(real_image, tf.float32)     #Because CNN expect(batch,H,W,D,CHANNEL)
    return input_image,real_image


@tf.function
def decode_lowerONLY_ct(Serialized_example):
    features = {
                'covid_lbl': tf.io.FixedLenFeature([],tf.int64),
                'lower_img': tf.io.FixedLenFeature([],tf.string),
                'high_img':tf.io.FixedLenFeature([],tf.string),
                'Patch_h': tf.io.FixedLenFeature([],tf.int64), #h
                'Patch_w': tf.io.FixedLenFeature([],tf.int64), #w
                'Sub_id': tf.io.FixedLenFeature([],tf.string)
                                }

    examples    = tf.io.parse_single_example(Serialized_example,features)
    img_shape   = [256,256]
    ##Decode_image_float
    image_1     = tf.io.decode_raw(examples['lower_img'], float) ##Decode_image_float
    input_image = tf.reshape(image_1,img_shape)        #Resgapping_the_data
    input_image = tf.expand_dims(input_image, axis=-1) #Because CNN expect(batch,H,W,D,CHANNEL)
    input_image = tf.cast(input_image, tf.float32)     #Because CNN expect(batch,H,W,D,CHANNEL)


    image_2    = tf.io.decode_raw(examples['high_img'], float) ##Decode_image_float
    real_image = tf.reshape(image_2,img_shape)        #Resgapping_the_data
    real_image = tf.expand_dims(real_image, axis=-1) #Because CNN expect(batch,H,W,D,CHANNEL)
    real_image = tf.cast(real_image, tf.float32)     #Because CNN expect(batch,H,W,D,CHANNEL)
    return input_image,real_image



@tf.function
def decode_medium_ct(Serialized_example):
    features = {
                'covid_lbl': tf.io.FixedLenFeature([],tf.int64),
                'lower_img': tf.io.FixedLenFeature([],tf.string),
                'medium_img':tf.io.FixedLenFeature([],tf.string),
                'high_img':tf.io.FixedLenFeature([],tf.string),
                'Patch_h': tf.io.FixedLenFeature([],tf.int64), #h
                'Patch_w': tf.io.FixedLenFeature([],tf.int64), #w
                'Sub_id': tf.io.FixedLenFeature([],tf.string)
                                }

    examples    = tf.io.parse_single_example(Serialized_example,features)
    img_shape   = [256,256]
    ##Decode_image_float
    image_1     = tf.io.decode_raw(examples['medium_img'], float) ##Decode_image_float
    input_image = tf.reshape(image_1,img_shape)        #Resgapping_the_data
    input_image = tf.expand_dims(input_image, axis=-1) #Because CNN expect(batch,H,W,D,CHANNEL)
    input_image = tf.cast(input_image, tf.float32)     #Because CNN expect(batch,H,W,D,CHANNEL)


    image_2    = tf.io.decode_raw(examples['high_img'], float) ##Decode_image_float
    real_image = tf.reshape(image_2,img_shape)        #Resgapping_the_data
    real_image = tf.expand_dims(real_image, axis=-1) #Because CNN expect(batch,H,W,D,CHANNEL)
    real_image = tf.cast(real_image, tf.float32)     #Because CNN expect(batch,H,W,D,CHANNEL)
    return input_image,real_image
