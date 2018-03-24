# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 21:08:38 2018

@author: pc
"""
import os
import _pickle as pk
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding , Activation ,Dense,Dropout
from keras.layers import LSTM,GRU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras import backend as K
from keras.models import Model
from keras import regularizers

train_path = './data/training_label.txt'
self_path = './data/training_nolabel.txt'
MAX_vocab_size = int(20000)
MAX_length = int(100)
save_path = './token/'
model_name = './model/model.h5'
embedding_size = 100
val_ratio = float(0.1)
nb_epoch = int(30)
batch_size = 128
action = 'semi'
loss_function = 'binary_crossentropy'
threshold = 0.7

class DataManger:
    def __init__(self):
        self.data = {}
    def add_data(self,name,data_path,with_label = True):
        print('read data for, %s ...'%data_path)
        X,Y = [],[]
        with open(data_path,'r',encoding="utf-8") as f:
            for line in f:
                if with_label:
                    lines = line.strip().split(' +++$+++ ')
                    X.append(lines[1])
                    Y.append(int(lines[0]))
                else:
                    X.append(line)
            if with_label:
                self.data[name] = [X,Y]
            else : 
                self.data[name] = [X]
    def tokenizer(self,vocab_size):
        print ('create new tokenizer')
        self.tokenizer = Tokenizer(num_words=vocab_size)
        for key in self.data:
            print ('tokenizing %s'%key)
            texts = self.data[key][0]
            self.tokenizer.fit_on_texts(texts)
    # Save tokenizer to specified path
    def save_tokenizer(self, path):
        print ('save tokenizer to %s'%path)
        pk.dump(self.tokenizer, open(path, 'wb'))
            
    # Load tokenizer from specified path
    def load_tokenizer(self,path):
        print ('Load tokenizer from %s'%path)
        self.tokenizer = pk.load(open(path, 'rb'))
    # Convert words in data to index and pad to equal size
    #  maxlen : max length after padding
    def to_sequence(self, maxlen):
        self.maxlen = maxlen
        for key in self.data:
            print ('Converting %s to sequences'%key)
            tmp = self.tokenizer.texts_to_sequences(self.data[key][0])
            #tmp = pad_sequences(tmp, maxlen=maxlen)
            self.data[key][0] = np.array(pad_sequences(tmp, maxlen=maxlen))
    
    # Convert texts in data to BOW feature
    def to_bow(self):
        for key in self.data:
            print ('Converting %s to tfidf'%key)
            self.data[key][0] = self.tokenizer.texts_to_matrix(self.data[key][0],mode='count')
    
    # Convert label to category type, call this function if use categorical loss
    def to_category(self):
        for key in self.data:
            if len(self.data[key]) == 2:
                self.data[key][1] = np.array(to_categorical(self.data[key][1]))
    def get_semi_data(self,name,label,threshold,loss_function) : 
        # if th==0.3, will pick label>0.7 and label<0.3
        label = np.squeeze(label)
        index = (label>1-threshold) + (label<threshold)
        semi_X = self.data[name][0]
        semi_Y = np.greater(label, 0.5).astype(np.int32)
        if loss_function=='binary_crossentropy':
            return semi_X[index,:], semi_Y[index]
        elif loss_function=='categorical_crossentropy':
            return semi_X[index,:], to_categorical(semi_Y[index])
        else :
            raise Exception('Unknown loss function : %s'%loss_function)
    # get data by name
    def get_data(self,name):
        return self.data[name]

    # split data to two part by a specified ratio
    #  name  : string, same as add_data
    #  ratio : float, ratio to split
    def split_data(self, name, ratio):
        data = self.data[name]
        X = data[0]
        Y = data[1]
        data_size = len(X)
        val_size = int(data_size * ratio)
        return (X[val_size:],Y[val_size:]),(X[:val_size],Y[:val_size])

def get_embedding_dict(path):
  embedding_dict = {}
  with open(path, 'r',encoding = 'utf-8') as f:
    for line in f:
      values = line.split(' ')
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embedding_dict[word] = coefs
    return embedding_dict   

def get_embedding_matrix(word_index, embedding_dict, num_words, embedding_dim):
  embedding_matrix = np.zeros((num_words,embedding_dim))
  
  for word, i in word_index.items():
    if i < num_words:
      embedding_vector = embedding_dict.get(word)
      if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
  return embedding_matrix

def main():
    dm = DataManger()
    print('Loading data')
    if action == 'train':
        dm.add_data('train',train_path,True) 
    elif action == 'semi':
        dm.add_data('train',train_path,True)
        dm.add_data('semi',self_path,False)
    
    print ('get Tokenizer...')
    dm.tokenizer(MAX_vocab_size)  
    word_index = dm.tokenizer.word_index
    
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if not os.path.exists(os.path.join(save_path,'token.pk')):
        dm.save_tokenizer(os.path.join(save_path,'token.pk'))    
    dm.to_sequence(MAX_length ) 
    
    
    if os.path.exists(model_name):
        print('Loading model...')
        model = load_model(model_name)
    else: 
        ### get mebedding matrix from glove
        print ('Get embedding dict from glove.')
        embedding_dict=get_embedding_dict('./glove/glove.6B.{}d.txt'.format(embedding_size))
        print ('Found {} word vectors.'.format(len(embedding_dict)))
        num_words = len(word_index) + 1
        print ('Create embedding matrix.')
        embedding_matrix = get_embedding_matrix(word_index, embedding_dict, num_words, embedding_size)
    
    print ('Building model.')
    model = Sequential()
    model.add(Embedding(num_words,
                        embedding_size,
                        weights=[embedding_matrix],
                        input_length=MAX_length,
                        trainable=False))
    model.add(GRU(128))
    model.add(Activation('tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1,activation='sigmoid'))
    model.summary()  
    adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
    model.compile(loss=loss_function,
                  optimizer=adam,
                  metrics=[ 'accuracy',])
    if action == 'train':
        (X,Y),(X_val,Y_val) = dm.split_data('train',val_ratio)
        earlystopping = EarlyStopping(monitor='val_acc', patience = 10, verbose=1, mode='max')
        checkpoint = ModelCheckpoint('ckpt/weights.{epoch:03d}-{val_acc:.5f}.h5',
                                 verbose=1,
                                 save_best_only=True,
                                 monitor='val_acc',
                                 mode='max')

        hist = model.fit(np.asarray(X),np.asarray( Y),
                     validation_data=(np.asarray(X_val), np.asarray(Y_val)),
                     epochs=nb_epoch,
                     batch_size=batch_size,
                     callbacks=[earlystopping,checkpoint])
    elif action == 'semi':
        (X,Y),(X_val,Y_val) = dm.split_data('train',val_ratio)
        [semi_all_X] = dm.get_data('semi')
        earlystopping = EarlyStopping(monitor='val_acc', patience = 10, verbose=1, mode='max')
        checkpoint = ModelCheckpoint('ckpt/weights.{epoch:03d}-{val_acc:.5f}.h5',
                                 verbose=1,
                                 save_best_only=True,
                                 monitor='val_acc',
                                 mode='max')
        for i in range(10):
            #label the semi_data
            semi_pred = model.predict(semi_all_X,batch_size=1024,verbose=True)
            semi_X,semi_Y = dm.get_semi_data('semi',semi_pred,threshold,loss_function)
            semi_X = np.concatenate((semi_X,X))
            semi_Y = np.concatenate((semi_Y,Y))
            print ('-- iteration %d  semi_data size: %d' %(i+1,len(semi_X)))
            hist = model.fit(np.asarray(semi_X),np.asarray(semi_Y), 
                                validation_data=(np.asarray(X_val), np.asarray(Y_val)),
                                epochs=2, 
                                batch_size=batch_size,
                                callbacks=[checkpoint, earlystopping] )
            
    ################################################
    # We need to save model & categories & tokenizer
    ################################################
    model.save(model_name)
    
if __name__ == '__main__':
    main()