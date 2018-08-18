# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 01:05:36 2018

@author: Heller
"""
import os
from tqdm import tqdm
from random import shuffle
import numpy as np
import cv2
train='train'
test='test'
lr=0.0003
modeln='myfirstmodel'
def label_img(label):
    if label == 'rose': return [1,0,0,0,0]
    elif label == 'dandelion': return [0,1,0,0,0]
    elif label =='tulip':return [0,0,1,0,0]
    elif label== 'daisy':return [0,0,0,1,0]
    elif label=='sunflower':return[0,0,0,0,1]
def return_training():
    training_data=[]
    for i in tqdm(os.listdir(train)):
        label=label_img(i)
        path=os.path.join(train,i)
        for j in tqdm(os.listdir(path)):
            path2=os.path.join(path,j)
            img=cv2.resize(cv2.imread(path2,cv2.IMREAD_GRAYSCALE),(50,50))
            training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('training.npy',training_data)
    return training_data
def return_testing():
    testing_data=[]
    for i in tqdm(os.listdir(test)):
        label=label_img(i)
        path=os.path.join(test,i)
        for j in tqdm(os.listdir(path)):
            path2=os.path.join(path,j)
            img=cv2.resize(cv2.imread(path2,cv2.IMREAD_GRAYSCALE),(50,50))
            testing_data.append([np.array(img),np.array(label)])
    shuffle(testing_data)
    np.save('testing.npy',testing_data)
    return testing_data
traindata=return_training()
testdata=return_testing()
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
convnet = input_data(shape=[None, 50, 50, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 5, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')
if os.path.exists('{}.meta'.format(modeln)):
    model.load(modeln)
print (traindata)
x=np.array([i[0] for i in traindata]).reshape(-1,50,50,1)
y=[i[1] for i in traindata]
tx=np.array([i[0] for i in testdata]).reshape(-1,50,50,1)
ty=[i[1] for i in testdata]
model.fit({'input': x}, {'targets': y}, n_epoch=3, validation_set=({'input': tx}, {'targets': ty}), 
    snapshot_step=500, show_metric=True, run_id=modeln)
model.save(modeln)