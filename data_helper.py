#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 09:32:11 2019

@author: mahbubcseju
"""
import csv
import numpy as np

class DataHelper:
    def __init__(self, data_path):
        self.data_path = data_path
        self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
        self.num_class= 4
        self.input_length = 1014
    
    def read_data(self):
        
        one_hot = np.eye(self.num_class, dtype='int64')
        x_data = []
        y_data = []
        
        dict_char = {}
        for idx, char in enumerate(self.alphabet):
            dict_char[char] = idx + 1
        cnt = 0
        with open(self.data_path,'r') as file:
            reader = csv.reader(file, delimiter = ',', quotechar = '"')
            
            for row in reader:
                cnt = cnt + 1
                if cnt == 12000:
                    break
                one_hot_index = int(row[0])-1
                y_data.append(one_hot[one_hot_index])
                text= []
                for col in row[1:]:
                    col = col.lower()
                    for char in col:
                        if char  in self.alphabet:
                            text.append(char)
                        else:
                            text.append(' ')
                final_text = []
                removed_text = []
                for i in range(0,len(text)):
                    if (i> 0 and text[i] == text[i-1] and text[i] == ' '):
                        removed_text.append(text[i])
                    else:
                        final_text.append(text[i])
                np_data = np.zeros(self.input_length, dtype='int64')
                
                for i in range(0,min(self.input_length,len(final_text))):
                    if final_text[i] in dict_char:
                        np_data[i] = dict_char[final_text[i]]
                x_data.append(np_data)
                
        return np.asarray(x_data), np.asarray(y_data)