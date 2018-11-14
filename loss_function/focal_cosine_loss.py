#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 20:52:41 2018

@author: ws
"""

# define focal_cosine_loss
import keras.backend as K
import tensorflow as tf

def focal_cosine_loss(y_true,y_pred,s=10.0,m=0.35,gamma=2,alpha=3.0):
  # get logits and labels
  Nums = y_true.get_shape()[0]
  y_true = y_true[0:Nums,...]
  y_pred = y_pred[0:Nums,...]
  # target margin
  target_margin = m*y_true
  # target logits
  target_logits = s*(y_pred - target_margin)
  # epsilon
  epsilon = 1e-12
  # labels
  labels = tf.cast(K.argmax(y_true,axis=-1),dtype=tf.float32)
  # target_probs
  target_probs = K.softmax(target_logits)
  neg_probs,pos_probs = target_probs[:,0],target_probs[:,1]
  # compute pos loss
  pos_loss = -K.mean(labels*K.pow(1.0-pos_probs,gamma)*K.log(pos_probs+epsilon))
  # compute neg loss
  neg_loss = -alpha*K.mean((1.0-labels)*K.pow(1-neg_probs,gamma)*K.log(neg_probs+epsilon))
  loss = pos_loss + neg_loss
  return loss
  

if __name__ == '__main__':
  xs = tf.random_normal(shape=[128,2])
  ys = tf.zeros_like(xs)
  loss = focal_cosine_loss(ys,xs)
  
  