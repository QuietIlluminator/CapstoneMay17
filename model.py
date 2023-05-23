import zipfile
import io
import requests
from os import path
import os.path
import cv2
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from hypopt import GridSearch
from tqdm.notebook import tqdm
from keras.applications.mobilenet import MobileNet
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from copy import deepcopy
import struct
from keras.models import Model
from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
import numpy as np
import argparse
from PIL import Image
import random
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape
from keras.layers import Dense, Conv2D
from keras.models import Sequential
from tensorflow.keras.layers import *
from keras import backend as K
import keras
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import time
import os
print("Installing packages...")

print("Importing stuff...")


start_time = time.time()


X = np.load("X.npy")
X_g = np.load("X_g.npy")
y = np.load("y.npy")


# IMG_WIDTH = 100
# IMG_HEIGHT = 75

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101)
# X_g_train, X_g_test, y_train, y_test = train_test_split(X_g, y, test_size=0.4, random_state=101)

# label_dict = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
# y_train = [label_dict[label] for label in y_train]
# y_test = [label_dict[label] for label in y_test]

# Convert the labels to one-hot encoded vectors
# y_train = to_categorical(y_train, num_classes=7)
# y_test = to_categorical(y_test, num_classes=7)

# for i in (range(len(X_train))):
#   transform = random.randint(0,1)
#   if (transform == 0):
#     X_augmented.append(cv2.flip(X_train[i],1))
#     X_g_augmented.append(cv2.flip(X_g_train[i],1))
#     y_augmented.append(y_train[i])
#   else:
#     zoom = 0.33
#
#     centerX,centerY=int(IMG_HEIGHT/2),int(IMG_WIDTH/2)
#     radiusX,radiusY= int((1-zoom)*IMG_HEIGHT*2),int((1-zoom)*IMG_WIDTH*2)
#
#     minX,maxX=centerX-radiusX,centerX+radiusX
#     minY,maxY=centerY-radiusY,centerY+radiusY
#
#     cropped = (X_train[i])[minX:maxX, minY:maxY]
#     new_img = cv2.resize(cropped, (IMG_WIDTH, IMG_HEIGHT))
#     X_augmented.append(new_img)
#
#     cropped = (X_g_train[i])[minX:maxX, minY:maxY]
#     new_img = cv2.resize(cropped, (IMG_WIDTH, IMG_HEIGHT))
#     X_g_augmented.append(new_img)
#
#     y_augmented.append(y_train[i])
#
# X_augmented = np.array(X_augmented)
# X_g_augmented = np.array(X_g_augmented)
#
# y_augmented = np.array(y_augmented)

# X_train = np.vstack((X_train,X_augmented))
# X_g_train = np.vstack((X_g_train,X_g_augmented))

# y_train = np.append(y_train,y_augmented)

# print(X_train.shape)
# print(X_g_train.shape)


epochs = 20
batch_size = 10
layers = 5
dropout = 0.5
activation = 'relu'


model = Sequential()
for i in range(layers):
  model.add(Conv2D(32, (3, 3), padding='same'))
  model.add(Activation(activation))

model.add(Conv2D(64, (3, 3)))
model.add(Activation(activation))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(dropout / 2.0))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation(activation))
model.add(Conv2D(128, (3, 3)))
model.add(Activation(activation))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(dropout / 2.0))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation(activation))
model.add(Dropout(dropout))
model.add(Dense(7))
model.add(Activation('softmax'))

opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


y_train_onehot = np.zeros((y_train.size, y_train.max().astype(int)+1))
y_train_onehot[np.arange(y_train.size), y_train.astype(int)] = 1

y_test_onehot = np.zeros((y_test.size, y_test.max().astype(int)+1))
y_test_onehot[np.arange(y_test.size), y_test.astype(int)] = 1

model.fit(X_train.astype(np.float32), y_train_onehot.astype(np.float32),
          validation_data=(X_test.astype(np.float32), y_test_onehot.astype(np.float32)), verbose=1, epochs=20)

# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=32)

model.save("my_model.h5")
