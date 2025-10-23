# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:02:35 2020

@author: jaehooncha

@email: chajaehoon79@gmail.com
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Concatenate, Dropout
from tensorflow.keras.models import Model

tf.keras.backend.set_floatx('float32')


def dense_bat_relu(x, node):
    x = Dense(node)(x)
    x = BatchNormalization(axis=-1, epsilon=1.001e-5)(x)
    x = Activation('relu')(x)
    return x


def branches(x, node):
    xu = dense_bat_relu(x, node)
    xd = dense_bat_relu(x, node)
    return xu, xd


class HADNN1(Model):
    def __init__(self, n_classes, input_dims):
        super(HADNN1, self).__init__()
        self.n_classes = n_classes  # 2 for x, y coordinates
        self.input_dims = input_dims

        self.fc = Dense(self.n_classes)
        self.drop = Dropout(0.3)

    def build(self):
        self.inputs = tf.keras.layers.Input(shape=self.input_dims)
        x = self.inputs
        
        x = dense_bat_relu(x, 128)
        x = self.drop(x)
        
        x1u, x1d = branches(x, 64)
        
        x = dense_bat_relu(x1u, 64)
        c = Concatenate()([x1d, x])
        
        self.pred = self.fc(c)
        
        return tf.keras.Model(inputs=self.inputs, outputs=self.pred)