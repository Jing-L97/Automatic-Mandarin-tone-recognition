# -*- coding: utf-8 -*-
"""
Pitch-based classifier
@author: Crystal

"""
from urllib.parse import quote
import glob
import numpy as np
import pandas as pd
from pandas import DataFrame
import parselmouth
from parselmouth.praat import call
import math
from pylab import *
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

'''
# get pitch contour(get rid of zero)
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
        #filler_number = 10 - len(raw_result)
        #filler = pitch_values[len(pitch_values)-filler_number]
        filler_number = len(raw_result)-10
        filler = pitch_values[filler_number]
        result = np.append(raw_result, filler)
    else:
        result = raw_result
    # get 五度值   Value =5* (log(x)-log(min))/(log(max)-log(min))
    maxPitch = np.amax(pitch_values)
    minPitch = np.amin(pitch_values)
    unified_pitch_lst = []
    for i in result:
        unified_pitch = 5* (log10(i)-log10(minPitch))/(log10(maxPitch)-log10(minPitch))
        floated_pitch = float(unified_pitch)
        unified_pitch_lst.append(floated_pitch)
    pitch=DataFrame(unified_pitch_lst)
#Q: do I need to further depict the trend?
    return pitch
'''
# get pitch contour(get rid of zero)
def pitchContour(voiceID, f0min, f0max):
    sound = parselmouth.Sound(voiceID) 
    pitch = call(sound, "To Pitch", 0.0,f0min,f0max)
    raw_pitch_values = pitch.selected_array['frequency']
    pitch_values = raw_pitch_values[raw_pitch_values != 0]
    interval = math.ceil(len(pitch_values)/10)
    raw_result = [sum(x)/len(x) for x in (pitch_values[k:k+interval] for k in range(0,len(pitch_values),interval))]
    # To ensure to get 10 averaged pitch values
    if len(raw_result) > 10:
        redundent = len(raw_result)-10
        result = raw_result[redundant:len(raw_result)]
    elif len(raw_result) < 10:
        filler_number = len(raw_result)-10
        filler = pitch_values[filler_number]
        result = np.append(raw_result, filler)
    else:
        result = raw_result
    # get 五度值   Value =5* (log(x)-log(min))/(log(max)-log(min))
    maxPitch = np.amax(pitch_values)
    minPitch = np.amin(pitch_values)
    unified_pitch_lst = []
    for i in result:
        unified_pitch = 5* (log10(i)-log10(minPitch))/(log10(maxPitch)-log10(minPitch))
        floated_pitch = float(unified_pitch)
        unified_pitch_lst.append(floated_pitch)
    pitch=DataFrame(unified_pitch_lst)
    return pitch


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


# get pitch contour
def get_data(filepath):
    pitch = pd.DataFrame()
    labels_temp = []
    for f in os.listdir(filepath): 
        pitch_temp = pd.concat([pitch,pitchContour(filepath + '/' + f,75,600)],axis=1)
        pitch = pitch_temp
        labels_temp.append(f.split('_')[0][-1]) 
    pitch_final = pitch.T
    labels = pd.DataFrame(labels_temp)
    pitch_final = pitch_final.reset_index(drop= True)
    labels = labels.reset_index(drop= True)
    feature = pd.concat([labels,pitch_final],axis=1)
    feature.columns =['Tone','P1','P2','P3','P4','P5','P6','P7','P8','P9','P10']
    return feature

#get the data              
pitch_FV1 = get_data('./FV1')
#list_FV1 = pitch_FV1.index[pitch_FV1['P10'] == 'nan'].tolist()
pitch_FV2 = get_data('./FV2')
pitch_FV3 = get_data('./FV3')
pitch_MV1 = get_data('./MV1')
pitch_MV2 = get_data('./MV2')
pitch_MV3 = get_data('./MV3')

pitch_FV1['P10'].isnull().sum()
pitch_FV2['P10'].isnull().sum()
pitch_FV3['P10'].isnull().sum()

pitch_MV1['P10'].isnull().sum()
pitch_MV2['P10'].isnull().sum()
pitch_MV3['P10'].isnull().sum()

pitch_female = pd.concat([pitch_FV1, pitch_FV2, pitch_FV3],axis=0)
pitch_female = pitch_female.dropna()
pitch_male = pd.concat([pitch_MV1, pitch_MV2, pitch_MV3],axis=0)
pitch_male = pitch_male.dropna()
pitch_all = pd.concat([pitch_female, pitch_male],axis=0)
label_all = pitch_all['Tone']
label_female = pitch_female['Tone']
label_male = pitch_male['Tone']
feature_female = pitch_female.drop('Tone', 1)
feature_male = pitch_male.drop('Tone', 1)
feature_all = pitch_all.drop('Tone', 1)

#train the Random Forest Classifier
feature_train, feature_test, target_train, target_test = train_test_split(feature_all, label_all, test_size=0.5,random_state=1)
clf = RandomForestClassifier()
clf.fit(feature_train,target_train)
predict_results=clf.predict(feature_test)
print(accuracy_score(predict_results, target_test))

conf_mat = confusion_matrix(target_test, predict_results)
print(conf_mat)
print(classification_report(target_test, predict_results))

#train on female and test on male using the Random Forest Classifier
feature_train = feature_female
feature_test = feature_male
target_train = label_female 
target_test = label_male 
clf = RandomForestClassifier()
clf.fit(feature_train,target_train)
predict_results=clf.predict(feature_test)
print(accuracy_score(predict_results, target_test))

conf_mat = confusion_matrix(target_test, predict_results)
print(conf_mat)
print(classification_report(target_test, predict_results))

#train on male and test on female using the Random Forest Classifier
feature_train = feature_male
feature_test = feature_female
target_train = label_male 
target_test = label_female 
clf = RandomForestClassifier()
clf.fit(feature_train,target_train)
predict_results=clf.predict(feature_test)
print(accuracy_score(predict_results, target_test))

conf_mat = confusion_matrix(target_test, predict_results)
print(conf_mat)
print(classification_report(target_test, predict_results))




