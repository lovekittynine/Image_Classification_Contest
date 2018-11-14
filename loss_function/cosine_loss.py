#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 22:56:37 2018

@author: ws
"""

# define cosine-softmax loss

import keras.backend as K


def cosine_softmax_loss(y_true,y_pred,s=10.0,margin=0.35):
    """
    S: radius of sphere default 10.0
    margin: default 0.35
    """
    # get current class margin
    target_margin = margin*y_true
    # get target logits
    target_logits = s*(y_pred - target_margin)
    # compute softmax loss
    loss = K.categorical_crossentropy(y_true,target_logits,from_logits=True)
    return loss
    