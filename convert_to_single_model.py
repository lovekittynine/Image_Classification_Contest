#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 14:39:47 2018

@author: admin3
"""

# convert multi-gpu model to single GPU-model

from CNN_model.multi_model import build_siamese_model
from keras.models import Model
import os
import tensorflow as tf
from keras.utils.training_utils import multi_gpu_model

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

with tf.device('/cpu:0'):
  model = build_siamese_model(training=False)
# restore multi-gpu model
parallel_model = multi_gpu_model(model,gpus=2)
# load weights
ckpt = os.path.join('checkpoint2','model-45-1.000.h5')
parallel_model.load_weights(ckpt,by_name=True)
# save single gpu model
model.save_weights(os.path.join('checkpoint2','single-model-45-1.000.h5'))
