#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 21:14:53 2018

@author: wsw
"""

# build model
from .resnet50 import ResNet50
from keras.layers import Dense,Dropout,Concatenate
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Subtract
from keras.layers.advanced_activations import LeakyReLU
import keras.backend as K
from keras.constraints import unitnorm
from keras.layers import Lambda
import tensorflow as tf


def build_model(input_shape=[224,224,3],num_class=6):
  # construct model with CBAM
  resnet50 = ResNet50(include_top=False,
                      input_shape=input_shape,
                      weights=None,
                      pooling='avg')
  # extract a batch features
  features = resnet50.output
  # l2 normalize for features
  features = Lambda(lambda x:K.l2_normalize(features,axis=-1))(features)
  # get class probs output
  probs_out1 = Dense(num_class,
                     activation=None,
                     name='cls',
                     kernel_constraint=unitnorm(axis=0),
                     kernel_regularizer=l2(1e-4),
                     use_bias=False)(features)
  # split features to feature1 and feature2
  feature1,feature2 = Lambda(lambda x:tf.split(features,2,axis=0))(features)
  # feature difference learn similarity
  feat_diff = Subtract()([feature1,feature2])
  # in order to main batchsize shape
  constant = Lambda(lambda x:K.zeros_like(x))(feat_diff)
  feat_diff = Concatenate(axis=0)([feat_diff,constant])
  
  x = Dense(units=128,activation=None)(feat_diff)
  x = LeakyReLU(alpha=0.1)(x)
  x = Dropout(rate=0.8)(x)
  similarity_out = Dense(units=2,
                         activation=None,
                         name='similarity',
                         kernel_regularizer=l2(1e-4),
                         kernel_constraint=unitnorm(axis=0),
                         use_bias=False)(x)
  # construct model
  model = Model(inputs=resnet50.input,outputs=[probs_out1,similarity_out])
  return model


if __name__ == '__main__':
  model = build_model()