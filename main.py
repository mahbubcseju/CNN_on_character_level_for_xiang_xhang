#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 09:20:13 2019

@author: mahbubcseju
"""

from data_helper import DataHelper
from models import Models
input_length = 1014
vocab_size = 70
embedding_dim = 128
conv_layer_info= [
        [64,7,3],
        [64,7,3],
        [64,3],
        [64,3],
        [64,3],
        [64,3,3]
        ]
conncted_layer_info = [
        [256,0.5],
        [256, 0.5]
        ]
batch_size = 128
epochs = 10
num_of_classes = 4
data = DataHelper("data-ag-news/train.csv")

x_train, y_train = data.read_data()

data1 = DataHelper("data-ag-news/test.csv")

x_test, y_test = data.read_data()

model = Models(x_train, y_train, x_test, y_test, num_of_classes)
model.models( input_length, vocab_size, embedding_dim, conv_layer_info, conncted_layer_info,batch_size,epochs)


