#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 16:27:09 2018

@author: sw
"""

# train mutil model

import os
from CNN_model.multi_model import build_siamese_model
from MyImageGenerator import MyImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam,SGD
from keras.callbacks import ReduceLROnPlateau,LearningRateScheduler
import math
from loss_function.cosine_loss import cosine_softmax_loss
import keras.backend as K
import tensorflow as tf
from keras.utils.training_utils import multi_gpu_model


class ParallelModelCheckpoint(ModelCheckpoint): 
  def __init__(self,
               model,
               filepath, 
               monitor='val_loss', 
               verbose=0,
               save_best_only=False,
               save_weights_only=False,
               mode='auto', 
               period=1): 
    self.single_model = model 
    
    super(ParallelModelCheckpoint,self).__init__(filepath, 
                                                 monitor, 
                                                 verbose,save_best_only, 
                                                 save_weights_only,
                                                 mode,
                                                 period) 
    def set_model(self, model): 
      super(ParallelModelCheckpoint,self).set_model(self.single_model)


# set argument parser
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_name','siamese','decide which model to use for train')
tf.app.flags.DEFINE_integer('checkpoint_num',888,'checkpoint number')
tf.app.flags.DEFINE_integer('batchsize',32,'train and evaluate batchsize')
tf.app.flags.DEFINE_integer('gpu_nums',2,'gpu numbers for train')

if FLAGS.gpu_nums == 1:
  # single GPU Training
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


train_datagen = MyImageDataGenerator(
                                   # preprocessing_function=preprocess_input,
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

valid_datagen = MyImageDataGenerator(
                                   # preprocessing_function=preprocess_input,
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
if FLAGS.model_name=='siamese':
  if FLAGS.gpu_nums==2:
    # creat multi GPU-model
    with tf.device('/cpu:0'):
      base_model = build_siamese_model(training=False)
      base_model.load_weights('./checkpoint2/final-epoch.h5')
    model = multi_gpu_model(base_model,gpus=FLAGS.gpu_nums)
  else:
    # creat single GPU-model
    base_model = build_siamese_model(training=False)
    base_model.load_weights('./checkpoint2/final-epoch.h5',by_name=True)
else:
  raise Exception('No model named %s'%FLAGS.model_name)
   
base_model.compile(optimizer=SGD(0.0001, momentum=0.9),
              loss={'cls_1':cosine_softmax_loss,
                             'cls_2':cosine_softmax_loss,
                             'feat_fuse':cosine_softmax_loss,
                             'logits':cosine_softmax_loss},     
              loss_weights={'cls_1':1.0,
                            'cls_2':1.0,
                            'feat_fuse':1.0,
                            'logits':1.0},
              metrics={'cls_1':'accuracy',
                       'cls_2':'accuracy',
                       'feat_fuse':'accuracy',
                       'logits':'accuracy'}) # categorical_crossentropy


weightname = '%s/model-{epoch:2d}-{val_feat_fuse_acc:.3f}.h5'%checkpoint

checkpointer = ModelCheckpoint(weightname, monitor='val_loss', verbose=1,
                               save_best_only=True, save_weights_only=True,
                               mode='auto', period=1)

# lrate1 = LearningRateScheduler(step_decay)
lrate = ReduceLROnPlateau(monitor='val_feat_fuse_acc',factor=0.5,patience=3,verbose=1)

base_model.fit_generator(
                    train_generator,
                    steps_per_epoch=train_generator.n//batchsize+1,
                    epochs=10,
                    validation_data=valid_generator,
                    validation_steps=valid_generator.n//batchsize+1,
                    verbose=1,
                    callbacks=[checkpointer,lrate])

# save final epoch
base_model.save_weights(os.path.join(checkpoint,'final-epoch.h5'))
