#!/usr/bin/env python
# -- coding: utf-8 --

import sys, os
import argparse
from keras.models import load_model
from termcolor import colored
from termcolor import cprint
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

model_name = "./ckpt/cnn_weights.002-0.70567.h5"
data_name = "./data/train.csv"
cmap_dir = "./image/cmap"
origin_dir = "./image/origin"
partial_see_dir = "./image/partial_see"

def read_data(filename,height=48,width=48):
    try:
        print('Loading X.npy & Y.npy')
        X = np.load('./data/X.npy')
        Y = np.load('./data/Y.npy')
    except:
        with open(filename,"r+") as f:
            line = f.read().strip().replace(',',' ').split('\n')[1:]
            raw_data = ' '.join(line)
            length = width * height +1
            data = np.array(raw_data.split()).astype('float').reshape(-1,length)
            X = data[:,1:]
            Y = data[:,0]
            
            X = X.reshape(-1,height,width,1)
            Y = Y.reshape(-1,1)
            print('Saving X.npy & Y.npy')
            np.save('./data/X.npy',X)
            np.save('./data/Y.npy',Y)
    return X,Y

def main():
    classifier = load_model(model_name)
    print (colored("Load model from {}".format(model_name),'red',attrs=['bold']))
    _X,Y = read_data(data_name)
    X = _X/255
    _X = _X.astype('int')
    Y = Y.squeeze()
    input_img = classifier.input
    img_ids = [-3002, -3004, -3005, -3006, -3009, -3010]
    
    for idx in img_ids:
        val_proba = classifier.predict(X[idx].reshape(-1,48,48,1))
        pred = val_proba.argmax(axis=-1)
        target = K.mean(classifier.output[:,pred])
        grads = K.gradients(target,input_img)[0]
        fn = K.function([input_img, K.learning_phase()], [grads])
        val_grads = fn([X[idx].reshape(-1, 48, 48, 1), 0])[0].reshape(48, 48, -1)
        val_grads *= -1
        val_grads = np.max(np.abs(val_grads), axis=-1, keepdims=True)

        # normalize
        val_grads = (val_grads - np.mean(val_grads)) / (np.std(val_grads) + 1e-30)
        val_grads *= 0.1
    
        # clip to [0, 1]
        val_grads += 0.5
        val_grads = np.clip(val_grads, 0, 1)
    
        # scale to [0, 1]
        val_grads /= np.max(val_grads)
    
        heatmap = val_grads.reshape(48, 48)
    
        print('ID: {}, Truth: {}, Prediction: {}'.format(idx, Y[idx], pred))
        # show original image
        plt.figure()
        plt.imshow(_X[idx].reshape(48, 48), cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(origin_dir, '{}.png'.format(idx)), dpi=100)
    
        thres = 0.5
        see = _X[idx].reshape(48, 48)
        see[np.where(heatmap <= thres)] = np.mean(see)
    
        plt.figure()
        plt.imshow(heatmap, cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(cmap_dir, '{}.png'.format(idx)), dpi=100)
    
        plt.figure()
        plt.imshow(see, cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(partial_see_dir, '{}.png'.format(idx)), dpi=100)
    
if __name__ == '__main__':
    main()