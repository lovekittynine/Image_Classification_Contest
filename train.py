#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 22:02:10 2018

@author: wsw
"""

# train model

import os
from CNN_model.model import build_resnet_model
#from keras.applications.resnet50 import ResNet50
from CNN_model.model import build_xception_model
from CNN_model.model import build_inception_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam,SGD
from keras.callbacks import ReduceLROnPlateau,LearningRateScheduler
import math
from loss_function.cosine_loss import cosine_softmax_loss
import keras.backend as K
import tensorflow as tf

# set argument parser
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_name','resnet50','decide which model to use for train')
tf.app.flags.DEFINE_integer('checkpoint_num',2,'checkpoint number')
tf.app.flags.DEFINE_integer('batchsize',28,'train and evaluate batchsize')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def step_decay(epoch):
  initial_lrate = 0.01
  drop = 0.5
  epochs_drop = 5.0
  lrate = initial_lrate* math.pow(drop, math.floor((1+epoch)/epochs_drop))
  print('Learning Rate Change to %.5f'%lrate)
  return lrate


# Train and validation ImageDataGenerator
batchsize = FLAGS.batchsize
trainDir = './trainDir'
validDir = './validDir'

# checkpoint dir
checkpoint = './checkpoint%d'%FLAGS.checkpoint_num
if not os.path.exists(checkpoint):
  os.mkdir(checkpoint)


train_datagen = ImageDataGenerator(
                                   #preprocessing_function=preprocess_input,
                                   rotation_range=90,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.2,
                                   zoom_range=0.15,
                                   channel_shift_range=0.1,
                                   vertical_flip=True,
                                   horizontal_flip=True,
                                   fill_mode='reflect',
                                   rescale=1./255,
                                   )

valid_datagen = ImageDataGenerator(
                                   #preprocessing_function=preprocess_input,
                                   channel_shift_range=0.1,
                                   vertical_flip=True,
                                   horizontal_flip=True,
                                   fill_mode='reflect',
                                   rescale=1./255,
                                   )


# make train generator
train_generator = train_datagen.flow_from_directory(
    trainDir,
    target_size=(224, 224),
    color_mode='rgb', # or 'grayscale'
    batch_size=batchsize,
    class_mode='categorical',
    )

valid_generator = valid_datagen.flow_from_directory(
    validDir,
    target_size=(224, 224),
    color_mode='rgb', # or 'grayscale'
    batch_size=batchsize,
    class_mode='categorical')





# build model
if FLAGS.model_name=='resnet50':
  ckpt = './pretrained_model/resnet50_model.h5'
  model = build_resnet_model()
elif FLAGS.model_name=='inceptionv3':
  ckpt = './pretrained_model/inception_v3_model.h5'
  model = build_inception_model()
elif FLAGS.model_name=='xception':
  ckpt = './pretrained_model/xception_model.h5'
  model = build_xception_model()
else:
  raise Exception('No model named %s'%FLAGS.model_name)
   
model.compile(optimizer=SGD(0.01, momentum=0.9),
              loss=cosine_softmax_loss, 
              metrics=['accuracy']) # categorical_crossentropy


# load model
if ckpt:
  model.load_weights(ckpt,skip_mismatch=True,by_name=True)


weightname = '%s/model-{epoch:2d}-{val_acc:.3f}.h5'%checkpoint

checkpointer = ModelCheckpoint(weightname, monitor='val_loss', verbose=1,
                               save_best_only=True, save_weights_only=True,
                               mode='auto', period=1)
lrate1 = LearningRateScheduler(step_decay)
lrate = ReduceLROnPlateau(monitor='val_acc',factor=0.5,patience=5,verbose=1)

model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.n//batchsize+1,
    epochs=50,
    validation_data=valid_generator,
    validation_steps=valid_generator.n//batchsize+1,
    callbacks=[checkpointer,lrate,lrate1])

# save final epoch
model.save_weights(os.path.join(checkpoint,'final-epoch.h5'))
