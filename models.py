#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 10:43:12 2019

@author: mahbubcseju
"""
from keras.layers import Input, Embedding, Convolution1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.models import Model

class Models:
    
    def __init__(self, x_train, y_train, x_test, y_test, num_of_classes):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.num_of_classes= num_of_classes
    def models(self, input_length, vocab_size, embedding_dim, conv_layer_info, connected_layer_info, batch_size,epochs):
        model_input = Input(shape = (input_length,), dtype = 'int64')
        model = Embedding(vocab_size, embedding_dim, input_length = input_length)(model_input)
        
        for i in range( len(conv_layer_info) ):
            model = Convolution1D(filters= conv_layer_info[i][0], kernel_size=conv_layer_info[i][1],
                         padding="valid",
                         activation="relu",
                         strides=1)(model)
            if len(conv_layer_info[i]) > 2:
                model = MaxPooling1D(pool_size = conv_layer_info[i][2])(model)
        
        model = Flatten()(model)
        
        for i in range(len(connected_layer_info)):
            model = Dense(connected_layer_info[i][0],activation = "relu")(model)
            model = Dropout(connected_layer_info[i][1])(model)
        model_output = Dense(self.num_of_classes, activation = "softmax")(model)
        model =Model(inputs=model_input,outputs=model_output)
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        print("Started Training : ")
        model.fit(self.x_train, self.y_train, batch_size = batch_size, epochs= epochs, validation_data=(self.x_test, self.y_test), 
                      verbose=2)
        print("Training Completed")
        
    