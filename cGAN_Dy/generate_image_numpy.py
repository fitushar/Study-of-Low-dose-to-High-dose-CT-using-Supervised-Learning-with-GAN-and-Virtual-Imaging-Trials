#import libraries
import tensorflow as tf
import os
import pathlib
import time
import datetime
import random
import pandas as pd
from tfrecords_utilities import*
from matplotlib import pyplot as plt
from config_pix2pix_2D_denoising import*
import pathlib

tf.config.optimizer.set_jit(False)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  except RuntimeError as e:
    # Visible devices must be set at program startup
    print(e)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)



#---Function to get the tfrecords list paths added and load the batched dataset
def getting_tfrecords_List(csv,path,csv_column_name):
    df = pd.read_csv(csv)
    df['tfrecords_with_path'] = path + df[csv_column_name]
    list_of_tfrecords = df['tfrecords_with_path'].tolist()
    list_of_tfrecords =random.sample(list_of_tfrecords, len(list_of_tfrecords))
    return list_of_tfrecords


def load_tfrecords_DualCT(record_mask_file,batch_size):
    dataset=tf.data.Dataset.list_files(record_mask_file).interleave(lambda x: tf.data.TFRecordDataset(x),cycle_length=NUMBER_OF_PARALLEL_CALL,num_parallel_calls=NUMBER_OF_PARALLEL_CALL)
    #dataset=dataset.map(decode_medium_ct,num_parallel_calls=NUMBER_OF_PARALLEL_CALL).take(len(record_mask_file)).cache(DATA_CACHE_TR).repeat(-1).batch(batch_size)
    dataset=dataset.map(decode_dual_ct,num_parallel_calls=NUMBER_OF_PARALLEL_CALL).repeat(-1).batch(batch_size)
    batched_dataset=dataset.prefetch(PARSHING)
    return batched_dataset


if  MODEL_TO_RUN=='Model_dual':
    tfrecords_train =  getting_tfrecords_List(csv=TRAIN_TFRECORDS_CSV,     path=TRAIN_TFRECORDS_PATH,     csv_column_name='tf_records_name')
    tfrecords_val   =  getting_tfrecords_List(csv=GENERATE_IMAGE_CSV,  path=TEST_TFRECORDS_PATH,csv_column_name='tf_records_name')

    train_batched_dataset = load_tfrecords_DualCT(tfrecords_train,BATCH_SIZE)
    val_batched_dataset   = load_tfrecords_DualCT(tfrecords_val,BATCH_SIZE)



#---Modeling
# This pix2pix is adopted ferom the implementation by the tenflow tutorials
# ref: https://www.tensorflow.org/tutorials/generative/pix2pix

#############----Build the generator
OUTPUT_CHANNELS = 1

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)
  result = tf.keras.Sequential()
  result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',kernel_initializer=initializer, use_bias=False))
  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())
  result.add(tf.keras.layers.LeakyReLU())
  return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,padding='same',kernel_initializer=initializer,use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
       result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result


def Generator():
  inputs = tf.keras.layers.Input(shape=[256, 256, 2])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

  x = inputs
  x = tf.keras.layers.Conv2D(filters=1,kernel_size=1,strides=(1, 1),padding='same',use_bias=False)(x)

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss


def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, 2], name='input_image')
  #inp = tf.keras.layers.Conv2D(filters=1,kernel_size=1,strides=(1, 1),padding='same',use_bias=False)(inp)
  tar = tf.keras.layers.Input(shape=[256, 256, 1], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
  down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
  down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)

generator = Generator()
discriminator = Discriminator()

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


checkpoint_dir = CHECKPOINT_DIR
pathlib.Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


log_dir=LOGDIR
pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(input_image, target, step):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
    tf.summary.scalar('disc_loss', disc_loss, step=step//1000)

pathlib.Path(GENERATE_IMAGE_DIR).mkdir(parents=True, exist_ok=True)
def generate_images(model, test_input, tar,step):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))
    display_list = [tar[0], prediction[0]]
    title = [ 'Ground Truth', 'Predicted Image']

    for i in range(2):
      plt.subplot(1, 2, i+1)
      plt.title(title[i])
      # Getting the pixel values in the [0, 1] range to plot.
      plt.imshow(display_list[i] ,cmap='gray')
      plt.axis('off')

    plt.savefig(GENERATE_IMAGE_DIR+'image_at_step_{:04d}.png'.format(step))

def validation_loop(model,test_ds):
    validation_psnr=[]
    validation_ssim=[]
    for (input_image, target) in test_ds.take(VALIDATION_DATASET_SIZE):
        prediction = model(input_image, training=True)
        #print('hi-pnsr')
        psnr = tf.image.psnr(target[0], prediction[0], max_val=1.0)
        #print('hi-ssim')
        ssim = tf.image.ssim(target[0], prediction[0], max_val=1.0, filter_size=11,filter_sigma=1.5, k1=0.01, k2=0.03)
        validation_psnr.append(psnr)
        validation_ssim.append(ssim)
    psnr_mean = np.mean(validation_psnr,axis=0)
    psnr_std = np.std(validation_psnr,axis=0)
    ssim_mean = np.mean(validation_ssim,axis=0)
    ssim_std = np.std(validation_ssim,axis=0)
    return psnr_mean,psnr_std,ssim_mean,ssim_std

def fit(train_ds, test_ds, steps):
  example_input, example_target = next(iter(test_ds.take(1)))
  start = time.time()

  psnr_mean_list =[]
  psnr_std_list  =[]
  ssim_mean_list =[]
  ssim_std_list  =[]
  step_list=[]

  for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
    if (step) % 1000 == 0:
      #display.clear_output(wait=True)

      if step != 0:
        print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

      start = time.time()

    train_step(input_image, target, step)

    # Training step
    if (step+1) % 10 == 0:
      print('.', end='', flush=True)


    # Save (checkpoint) the model every 5k steps
    if (step + 1) % 15000 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)
      generate_images(generator, example_input, example_target,step)
      psnr_mean,psnr_std,ssim_mean,ssim_std = validation_loop(generator,test_ds)
      print("validation-> psnr={:.2f},std={:.2f}; ssim={:.2f}-std={:.2f}".format(psnr_mean,psnr_std,ssim_mean,ssim_std))
      print(f"Step: {step//1000}k")
      psnr_mean_list.append(psnr_mean)
      psnr_std_list.append(psnr_std)
      ssim_mean_list.append(ssim_mean)
      ssim_std_list.append(ssim_std)
      step_list.append(step.numpy())
      with summary_writer.as_default():
          tf.summary.scalar('val_psnr', psnr_mean, step=step//15000)
          tf.summary.scalar('val_ssim', ssim_mean, step=step//15000)
  validation_df= pd.DataFrame(list(zip(step_list,psnr_mean_list,psnr_std_list,ssim_mean_list,ssim_std_list)),columns=['step','psnr_mean','psnr_std','ssim_mean','ssim_std'])
  validation_df.to_csv(VALIDATION_PSNR_SSIM_CSV_PATH, encoding='utf-8', index=False)

#fit(train_batched_dataset, val_batched_dataset, steps=TRAINING_DATASET_SIZE*100)


#fit(train_batched_dataset, val_batched_dataset, steps=TRAINING_DATASET_SIZE*100)

# Restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

def Testing_loop(model,test_ds):
    validation_psnr=[]
    validation_ssim=[]
    for (input_image, target) in test_ds.take(VALIDATION_DATASET_SIZE):
        prediction = model(input_image, training=True)
        #print('hi-pnsr')
        psnr = tf.image.psnr(target[0], prediction[0], max_val=1.0)
        #print('hi-ssim')
        ssim = tf.image.ssim(target[0], prediction[0], max_val=1.0, filter_size=11,filter_sigma=1.5, k1=0.01, k2=0.03)
        validation_psnr.append(psnr)
        validation_ssim.append(ssim)
    psnr_mean = np.mean(validation_psnr,axis=0)
    psnr_std = np.std(validation_psnr,axis=0)
    ssim_mean = np.mean(validation_ssim,axis=0)
    ssim_std = np.std(validation_ssim,axis=0)
    return psnr_mean,psnr_std,ssim_mean,ssim_std

psnr_mean,psnr_std,ssim_mean,ssim_std = Testing_loop(generator,val_batched_dataset)
print("validation-> psnr={:.2f},std={:.2f}; ssim={:.2f}-std={:.2f}".format(psnr_mean,psnr_std,ssim_mean,ssim_std))


def SAVE_ONE_TEST_GENERATEDIMAGE(model,test_ds):
    test_input, tar = next(iter(test_ds.take(1)))
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))
    display_list = [tar[0], prediction[0]]
    title = [ 'Ground Truth', 'Predicted Image']

    for i in range(2):
      plt.subplot(1, 2, i+1)
      plt.title(title[i])
      # Getting the pixel values in the [0, 1] range to plot.
      plt.imshow(display_list[i] ,cmap='gray')
      plt.axis('off')

    plt.savefig(ROOT_SAVING_PATHS+MODEL_TO_RUN+'/'+MODEL_TO_RUN+'_generated_test_image.png')
    np.save(ROOT_SAVING_PATHS+MODEL_TO_RUN+'/'+MODEL_TO_RUN+'_generated_test_image.npy',prediction[0])

SAVE_ONE_TEST_GENERATEDIMAGE(generator,val_batched_dataset)
