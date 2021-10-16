# -*- coding: utf-8 -*-
"""
ASR final codes

"""

import numpy as np
#import pandas as pd
import os
from matplotlib import pyplot as plt
import librosa
import librosa.display
import keras
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.utils import to_categorical
#import sklearn.metrics as metrics
import itertools
from sklearn.metrics import confusion_matrix
import parselmouth
from parselmouth.praat import call
from matplotlib import pyplot as plt
import math
from pylab import *

############################
#### 1 related funtions ####
############################

def mp3tomfcc(filepath):                                          
  audio, sample_rate = librosa.core.load(filepath)
  mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=60)
  max_pad = 60
  pad_width = max_pad - mfcc.shape[1]
  mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
  return mfcc

# get pitch contour
def pitchContour(voiceID, f0min, f0max):
    sound = parselmouth.Sound(voiceID) 
    pitch = call(sound, "To Pitch", 0.0,f0min,f0max)
    raw_pitch_values = pitch.selected_array['frequency']
    pitch_values = raw_pitch_values[raw_pitch_values != 0]
    interval = math.ceil(len(pitch_values)/10)
    raw_result = [sum(x)/len(x) for x in (pitch_values[k:k+interval] for k in range(0,len(pitch_values),interval))]
    # To ensure to get 10 averaged pitch values
    if len(raw_result) > 10:
        redundent = 10 - len(raw_result) 
        result = raw_result[:redundent]
    elif len(raw_result) < 10:
        filler_number = 10 - len(raw_result)
        filler = pitch_values[len(pitch_values)-filler_number]
        result = np.append(raw_result, filler)
    else:
        result = raw_result
    #   Value =5* (log(x)-log(min))/(log(max)-log(min))
    maxPitch = np.amax(pitch_values)
    minPitch = np.amin(pitch_values)
    unified_pitch_lst = []
    for i in result:
        unified_pitch = 5* (log10(i)-log10(minPitch))/(log10(maxPitch)-log10(minPitch))
        unified_pitch_lst.append(unified_pitch)
    #Q: do I need to further depict the trend?
    return unified_pitch_lst

# Combine pitch contour, jitter and shimmer 
def measurePitch(voiceID, f0min, f0max):
    sound = parselmouth.Sound(voiceID) 
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max)
    #create a praat pitch object
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    contour = pitchContour(voiceID, f0min, f0max)
    return localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, contour


def get_cnn_model(input_shape, num_classes):                                   
    model = Sequential()                                                      
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu',                
                     input_shape=input_shape))
    model.add(BatchNormalization())                                            
    model.add(Conv2D(48, kernel_size=(2, 2), activation='relu'))               
    model.add(BatchNormalization())
    model.add(Conv2D(120, kernel_size=(2, 2), activation='relu'))              
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))                                  
    model.add(Dropout(0.25))
    model.add(Flatten())                                                       
    model.add(Dense(128, activation='relu'))                                   
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))                                    
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))                        
    model.compile(loss=keras.losses.categorical_crossentropy,                  
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model

def plot_confusion_matrix(cm, classes,                                         
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)                         # plot the accuracy based on predicted tone labels
    plt.colorbar()                                                             # and and true tone labels
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True Tone')
    plt.xlabel('Predicted Tone')

   
def get_data(filepath):
    mfccs = []
    labels = []
    for f in os.listdir(filepath):
        if f.endswith('.mp3'):
            mfccs.append(mp3tomfcc(filepath + '/' + f))
            labels.append(f.split('_')[0][-1])  
    return np.asarray(mfccs), to_categorical(labels)

def feature_slice(mfcc_input, start, end):
    new_mfcc = mfcc_input[:,start:end]
    return new_mfcc

def data_training(mfcc_loaded, label_loaded):
    
    dim_1 = mfcc_loaded.shape[1]
    dim_2 = mfcc_loaded.shape[2]
    channels = 1
    classes = 5
    results = []

    X = mfcc_loaded
    X = X.reshape((mfcc_loaded.shape[0], dim_1, dim_2, channels))
    y = label_loaded
    input_shape = (dim_1, dim_2, channels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)  # Randomly allocate 50% of the data for training
    model = get_cnn_model(input_shape, classes)                                         # and 50% of the data for testing
    #model.summary()
    history = model.fit(X_train, y_train, batch_size=20, epochs=20, verbose=1,
                          validation_data=(X_test, y_test), class_weight=None)
    train_acc = model.evaluate(X_train, y_train, batch_size = 3, verbose = 1)    # evaluate model on training
    test_acc = model.evaluate(X_test, y_test, batch_size = 3, verbose = 1)       # evaluate model on testing

    y_pred = model.predict(X_test)    
    y_pred_labels = np.argmax(y_pred, axis=1)   
    y_test_labels = np.argmax(y_test, axis=1)   
    
    results = [train_acc, test_acc]
    
    cm = confusion_matrix(y_test_labels, y_pred_labels)
    np.set_printoptions(precision=2)

    plt.figure()
    plot_confusion_matrix(cm, classes=[1, 2, 3, 4],
                      title='Confusion Matrix')
    return results

##################################
#### 2 getting MFCCs & Labels ####
##################################

## data preparation
#file_path = './dataset'
mfcc_FV1, label_FV1 = get_data('./FV1')
mfcc_FV2, label_FV2 = get_data('./FV2')
mfcc_FV3, label_FV3 = get_data('./FV3')
mfcc_MV1, label_MV1 = get_data('./MV1')
mfcc_MV2, label_MV2 = get_data('./MV2')
mfcc_MV3, label_MV3 = get_data('./MV3')
np.save('./mfcc_FV1.npy', mfcc_FV1)                                            # Because it takes a certain amount of time to                        
np.save('./label_FV1.npy', label_FV1)
np.save('./mfcc_FV2.npy', mfcc_FV2)                                            # Because it takes a certain amount of time to                        
np.save('./label_FV2.npy', label_FV2)
np.save('./mfcc_FV3.npy', mfcc_FV3)                                            # Because it takes a certain amount of time to                        
np.save('./label_FV3.npy', label_FV3)
np.save('./mfcc_MV1.npy', mfcc_MV1)                                            # Because it takes a certain amount of time to                        
np.save('./label_MV1.npy', label_MV1)
np.save('./mfcc_MV2.npy', mfcc_MV2)                                            # Because it takes a certain amount of time to                        
np.save('./label_MV2.npy', label_MV2)
np.save('./mfcc_MV3.npy', mfcc_MV3)                                            # Because it takes a certain amount of time to                        
np.save('./label_MV3.npy', label_MV3)

mfcc_female = np.concatenate((mfcc_FV1, mfcc_FV2, mfcc_FV3), axis=0)
label_female = np.concatenate((label_FV1, label_FV2, label_FV3), axis=0)
mfcc_male = np.concatenate((mfcc_MV1, mfcc_MV2, mfcc_MV3), axis=0)
label_male = np.concatenate((label_MV1, label_MV2, label_MV3), axis=0)
mfcc_all = np.concatenate((mfcc_female, mfcc_male), axis=0)
label_all = np.concatenate((label_female, label_male), axis=0)
np.save('./mfcc_female.npy', mfcc_female)
np.save('./label_female.npy', label_female)
np.save('./mfcc_male.npy', mfcc_male)
np.save('./label_male.npy', label_male)
np.save('./mfcc_all.npy', mfcc_all)
np.save('./label_all.npy', label_all)

## data loading
mfcc_loaded = np.load('./mfcc_all.npy')
label_loaded = np.load('./label_all.npy')

finalresults = data_training(mfcc_loaded, label_loaded)
print(finalresults)


audio, sample_rate = librosa.core.load('a1_FV1_MP3.mp3')
mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=150)
    