#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 21:14:53 2018

@author: wsw
"""


# build model
from .resnet50 import ResNet50
# from .Xception import Xception
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense
from keras.models import Model
from keras.constraints import unitnorm
from keras.layers import Lambda,Input,Maximum,Concatenate
from keras.layers import GlobalAveragePooling2D
import keras.backend as K
from keras.layers import Add
import tensorflow as tf



def build_siamese_model(input_shape=[224,224,3],num_class=6,training=True):
  
  # construct model with CBAM
  resnet50 = ResNet50(include_top=False,
                      input_shape=input_shape,
                      weights=None,
                      pooling='avg')
  # load weights
  # base_model1
  if training:
      resnet50.load_weights('./pretrained_model/resnet50_model.h5',
                            by_name=True,
                            skip_mismatch=True)
  # construct model with CBAM
  
  xception = Xception(include_top=False,
                      weights=None,
                      input_shape=input_shape,
                      pooling='avg',
                      )
  if training:
      weights='./pretrained_model/xception_model.h5'
      xception.load_weights(weights)
      
  # inputs
  imgs = Input(shape=[224,224,3])
  # get features
  feature1 = resnet50(imgs)
  # print(feature1.shape)
  feature2 = xception(imgs)
  # print(feature2.shape)
  
  # get fusion features
  fusion_feature = Concatenate(axis=-1)([feature1,feature2])
  
  # apply global average pooling
  # feature1 = GlobalAveragePooling2D()(feature1)
  # feature2 = GlobalAveragePooling2D()(feature2)
  # fusion_feature = GlobalAveragePooling2D()(fusion_feature)
  
  # l2 normalize for features
  feature1 = Lambda(lambda x:K.l2_normalize(x,axis=-1))(feature1)
  feature2 = Lambda(lambda x:K.l2_normalize(x,axis=-1))(feature2)
  fusion_feature = Lambda(lambda x:K.l2_normalize(x,axis=-1))(fusion_feature)
  
  # get outputs
  output1 = Dense(units=num_class,
                  activation=None,
                  use_bias=False,
                  kernel_constraint=unitnorm(axis=0),
                  name='cls_1')(feature1)
  output2 = Dense(units=num_class,
                  activation=None,
                  use_bias=False,
                  kernel_constraint=unitnorm(axis=0),
                  name='cls_2')(feature2)
  output3 = Dense(units=num_class,
                  activation=None,
                  use_bias=False,
                  kernel_constraint=unitnorm(axis=0),
                  name='feat_fuse')(fusion_feature)
  # model fusion way1
  logits = Maximum(name='logits')([output1,output2,output3])
  # model fusion way2
#  logit1 = Lambda(lambda x:tf.multiply(0.2,x))(output1)
#  logit2 = Lambda(lambda x:tf.multiply(0.3,x))(output1)
#  logit3 = Lambda(lambda x:tf.multiply(0.5,x))(output1)
#  logits = Add(name='logits')([logit1,logit2,logit3])
  model = Model(inputs=imgs,outputs=[output1,output2,output3,logits])
  
  return model


if __name__ == '__main__':
  model = build_siamese_model(training=False)
