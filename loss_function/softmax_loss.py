#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 22:56:37 2018

@author: ws
"""

# define softmax loss

import keras.backend as K

def softmax_loss(y_true,y_pred):
    # compute softmax loss
    loss = K.categorical_crossentropy(y_true,y_pred,from_logits=True)
    return loss
