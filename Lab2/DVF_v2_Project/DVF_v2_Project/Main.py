#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:04:25 2022

@author: mvisintin

Sensor units are calibrated to acquire data at 25 Hz sampling 
frequency. 
The 5-min signals are divided into 5-sec segments so that 
480(=60x8) signal segments are obtained for each activity.

"""
from Scaler import scaler
from sklearn.cluster import KMeans
from MapRemap import mapping, remap
from Utils import multipage, Accuracy, Confusion_matrix
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


cm = plt.get_cmap('gist_rainbow')
line_styles = ['solid', 'dashed', 'dotted']
warnings.filterwarnings("ignore")

def generateDF(filedir, colnames, patients, activities, slices):
    x = pd.DataFrame()
    for pat in patients:
        for a in activities:
            subdir = 'a' + f"{a:02d}" + '/p' + str(pat) + '/'
            for s in slices:
                filename = filedir + subdir + 's' + f"{s:02d}" + '.txt'
                x1 = pd.read_csv(filename, usecols=sensors, names=colnames)
                x1['activity'] = a * np.ones((x1.shape[0],), dtype=int)
                x = pd.concat([x, x1], axis=0, join='outer', ignore_index=True,
                              keys=None, levels=None, names=None, verify_integrity=False,
                              sort=False, copy=True)
    return x


plt.close('all')
filedir = 'data/'
sensNames = [
    'T_xacc', 'T_yacc', 'T_zacc',
    'T_xgyro', 'T_ygyro', 'T_zgyro',
    'T_xmag', 'T_ymag', 'T_zmag',
    'RA_xacc', 'RA_yacc', 'RA_zacc',
    'RA_xgyro', 'RA_ygyro', 'RA_zgyro',
    'RA_xmag', 'RA_ymag', 'RA_zmag',
    'LA_xacc', 'LA_yacc', 'LA_zacc',
    'LA_xgyro', 'LA_ygyro', 'LA_zgyro',
    'LA_xmag', 'LA_ymag', 'LA_zmag',
    'RL_xacc', 'RL_yacc', 'RL_zacc',
    'RL_xgyro', 'RL_ygyro', 'RL_zgyro',
    'RL_xmag', 'RL_ymag', 'RL_zmag',
    'LL_xacc', 'LL_yacc', 'LL_zacc',
    'LL_xgyro', 'LL_ygyro', 'LL_zgyro',
    'LL_xmag', 'LL_ymag', 'LL_zmag']
actNames = [
    'sitting',  # 1
    'standing',  # 2
    'lying on back',  # 3
    'lying on right side',  # 4
    'ascending stairs',  # 5
    'descending stairs',  # 6
    'standing in an elevator still',  # 7
    'moving around in an elevator',  # 8
    'walking in a parking lot',  # 9
    'walking on a treadmill with a speed of 4 km/h in flat',  # 10
    'walking on a treadmill with a speed of 4 km/h in 15 deg inclined position',  # 11
    'running on a treadmill with a speed of 8 km/h',  # 12
    'exercising on a stepper',  # 13
    'exercising on a cross trainer',  # 14
    'cycling on an exercise bike in horizontal positions',  # 15
    'cycling on an exercise bike in vertical positions',  # 16
    'rowing',  # 17
    'jumping',  # 18
    'playing basketball'  # 19
]
actNamesShort = [
    'sitting',  # 1
    'standing',  # 2
    'lying.ba',  # 3
    'lying.ri',  # 4
    'asc.sta',  # 5
    'desc.sta',  # 6
    'stand.elev',  # 7
    'mov.elev',  # 8
    'walk.park',  # 9
    'walk.4.fl',  # 10
    'walk.4.15',  # 11
    'run.8',  # 12
    'exer.step',  # 13
    'exer.train',  # 14
    'cycl.hor',  # 15
    'cycl.ver',  # 16
    'rowing',  # 17
    'jumping',  # 18
    'play.bb'  # 19
]
ID = 307774
s = ID % 8 + 1
patients = [s]
activities = list(range(1, 20))
Num_activities = len(activities)
NAc = 19
actNamesSub = [actNamesShort[i - 1] for i in activities]
sensors = list(range(14))
sensNamesSub = [sensNames[i] for i in sensors]
Nslices = 30
Ntot = 60
slices = list(range(1, Nslices + 1))
testslices = list(range(Nslices + 1, Ntot))
fs = 25
samplesPerSlice = fs * 5
figures = []
for i in activities:
    activities = [i]
    x = generateDF(filedir, sensNamesSub, patients, activities, slices)
    x = x.drop(columns=['activity'])
    sensors = list(x.columns)
    data = x.values
    figures.append(plt.figure(figsize=(6, 6)))
    time = np.arange(data.shape[0]) / fs  # set the time axis
    for k in range(len(sensors)):
        lines = plt.plot(time, data[:, k], '.', label=sensors[k], markersize=1)
        lines[0].set_color(cm(k // 3 * 3 / len(sensors)))
        lines[0].set_linestyle(line_styles[k % 3])
    plt.legend()
    plt.grid()
    plt.xlabel('time (s)')
    plt.tight_layout()
    plt.title(actNames[i - 1])
centroids = np.zeros((NAc, len(sensors)))
stdpoints = np.zeros((NAc, len(sensors)))
fig1 = plt.figure(figsize=(12, 6))
figures.append(fig1)
for i in range(1, NAc + 1):
    activities = [i]
    x = generateDF(filedir, sensNamesSub, patients, activities, slices)
    x = x.drop(columns=['activity'])
    scaled_x = scaler(x)
    x = pd.DataFrame(scaled_x, columns=x.columns)
    centroids[i - 1, :] = x.mean().values
    plt.subplot(1, 2, 1)
    lines = plt.plot(centroids[i - 1, :], label=actNamesShort[i - 1])
    lines[0].set_color(cm(i // 3 * 3 / NAc))
    lines[0].set_linestyle(line_styles[i % 3])
    stdpoints[i - 1] = x.var().values
    plt.subplot(1, 2, 2)
    lines = plt.plot(stdpoints[i - 1, :], label=actNamesShort[i - 1])
    lines[0].set_color(cm(i // 3 * 3 / NAc))
    lines[0].set_linestyle(line_styles[i % 3])
plt.subplot(1, 2, 1)
plt.legend(loc='upper right')
plt.grid()
plt.title('Centroids using ' + str(len(sensors)) + ' sensors')
plt.xticks(np.arange(x.shape[1]), list(x.columns), rotation=90)
plt.subplot(1, 2, 2)
plt.legend(loc='upper right')
plt.grid()
plt.title('Standard deviation using ' + str(len(sensors)) + ' sensors')
plt.xticks(np.arange(x.shape[1]), list(x.columns), rotation=90)
plt.tight_layout()
d = np.zeros((NAc, NAc))
for i in range(NAc):
    for j in range(NAc):
        d[i, j] = np.linalg.norm(centroids[i] - centroids[j])

plt.matshow(d)
plt.colorbar()
plt.xticks(np.arange(NAc), actNamesShort, rotation=90)
plt.yticks(np.arange(NAc), actNamesShort)
plt.title('Between-centroids distance')
dd = d + np.eye(NAc) * 1e6
dmin = dd.min(axis=0)
dpoints = np.sqrt(np.sum(stdpoints ** 2, axis=1))
last_fig = plt.figure()
figures.append(last_fig)
plt.plot(dmin, label='minimum centroid distance')
plt.plot(dpoints, label='mean distance from points to centroid')
plt.grid()
plt.xticks(np.arange(NAc), actNamesShort, rotation=90)
plt.legend()
plt.tight_layout()
plt.show()
y_train = []
activities = list(range(1, 20))
x_train = generateDF(filedir, sensNamesSub, patients, activities, slices)
x_train = x_train.drop(columns=['activity'])
scaled_x = scaler(x_train)
x_train = pd.DataFrame(scaled_x, columns=x_train.columns)
x_test = generateDF(filedir, sensNamesSub, patients, activities, testslices)
y_test = x_test['activity']
x_test = x_test.drop(columns=['activity'])
scaled_x_test = scaler(x_test)
x_test = pd.DataFrame(scaled_x_test, columns=x_test.columns)
km = KMeans(n_clusters=19, init='k-means++', max_iter=300, n_init=10, random_state=0)
km.fit(x_train)
y_km = km.predict(x_test)
y_km_training = km.predict(x_train)
km_mapping = mapping(y_km)
km_mapping_training = mapping(y_km_training)
predicted = remap(km_mapping)
predicted_training = remap(km_mapping_training)
accuracy = Accuracy(activities, predicted)
accuracy_train = Accuracy(activities, predicted_training)
print('\nAccuracy on test set:\n' + str(accuracy))
print('\nAccuracy on train set:\n' + str(accuracy_train))
conf_matrix = Confusion_matrix(activities, predicted)
print('\nCofusion Matrix on test set:\n' + str(conf_matrix))
conf_matrix_train = Confusion_matrix(activities, predicted_training)
print('\nCofusion Matrix on train set:\n' + str(conf_matrix_train))
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.xlabel('Predicted_test')
plt.ylabel('Activities_test')
plt.show()
sns.heatmap(conf_matrix_train, annot=True, fmt='d')
plt.xlabel('Predicted_train')
plt.ylabel('Activities_train')
plt.show()
#plt.matshow(conf_matrix)
#plt.matshow(conf_matrix_train)
#plt.savefig('conf_matrix.png')
#plt.savefig('conf_matrix_train.png')
#plt.show()