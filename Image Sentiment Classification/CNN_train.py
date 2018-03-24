# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 22:17:36 2018

@author: acer
"""
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization
from keras import losses
from keras import optimizers
from keras.callbacks import CSVLogger, ModelCheckpoint

#Parameter
height = width = 48
num_classes = 7
input_shape = (height, width, 1)
batch_size = 128
epochs = 30
zoom_range = 0.2
model_name = './modle/cnn_model.h5'
isValid = 1

# Read the train data
with open('./data/train.csv',"r+") as f:
    line = f.read().strip().replace(',',' ').split('\n')[1:]
    raw_data = ' '.join(line)
    length = width*height+1
    data = np.array(raw_data.split()).astype('float').reshape(-1,length)
    X = data[:,1:]
    Y = data[:,0]
    X /= 255
    Y = Y.reshape(Y.shape[0],1)
    Y = to_categorical(Y,num_classes)    

X = X.reshape(X.shape[0],height,width,1)

if isValid:
    valid_num = 3000
    X_train,Y_train = X[:-valid_num],Y[:-valid_num]
    X_valid,Y_valid = X[-valid_num:],Y[-valid_num:]
    
else:
    X_train,Y_train =X,Y

# Construct the model
model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
model.add(LeakyReLU(alpha=0.03))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.03))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.03))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.3))

model.add(Conv2D(512, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.03))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()
#load model 
model.load_weights('ckpt/weights.002-0.71600.h5')
#Compole the model
model.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy'])

#Image PrePocessing
train_gen = ImageDataGenerator(rotation_range = 25,
                               width_shift_range = 0.1,
                               height_shift_range = 0.1,
                               shear_range = 0.1,
                               zoom_range = [1-zoom_range,1+zoom_range],
                               horizontal_flip = True)
train_gen.fit(X_train)

#Callbacks
callbacks = []
modelcheckpoint = ModelCheckpoint('ckpt/cnn_weights.{epoch:03d}-{val_acc:.5f}.h5', monitor='val_acc', save_best_only=True)
callbacks.append(modelcheckpoint)
csv_logger = CSVLogger('cnn_log.csv',separator=',',append=False)
callbacks.append(csv_logger)

#Fit the model 
if isValid:
    model.fit_generator(train_gen.flow(X_train,Y_train,batch_size=batch_size),
                        steps_per_epoch=10*X_train.shape[0]//batch_size,
                        epochs = epochs,
                        callbacks=callbacks,
                        validation_data=(X_valid,Y_valid))
else:
    model.fit_generator(train_gen.flow(X_train, Y_train, batch_size=batch_size),
                    steps_per_epoch=10*X_train.shape[0]//batch_size,
                    epochs=epochs,
                    callbacks=callbacks)
  
# Save modle
model.save(model_name)