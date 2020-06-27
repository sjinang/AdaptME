import numpy as np
import sklearn.decomposition
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
import pickle 
import dill
import tensorflow as tf
import joblib
from config import data_mir
from tensorflow.keras.regularizers import l1,l2
# import tensorflow.keras.backend as K

########################################################################
########################################################################

def RPA(y_true,y_pred):
    a = tf.abs(y_true-y_pred) < 0.5
    a = tf.cast(a,dtype=tf.float32)
    return 100*tf.reduce_mean(a)


def NN_regressor():
    model = Sequential()

    # model.add(Dropout(0, input_shape=(1025,)))
    n=256; data_mir.log=str(n); model.add(Dense(n, activation='tanh',input_dim=1025)) # ^softsign,tanh
    n=256; data_mir.log+='x'+str(n); model.add(Dense(n, activation='tanh'))
    n=256; data_mir.log+='x'+str(n); model.add(Dense(n, activation='tanh'))
    n=256; data_mir.log+='x'+str(n); model.add(Dense(n, activation='tanh'))
    # n=500; data_mir.log+='x'+str(n); model.add(Dense(n, activation='tanh',kernel_regularizer=keras.regularizers.l1()))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae',RPA]) 

    return model
    
########################################################################
########################################################################

def NN_classifier():
    model = Sequential()
    model.add(Dense(50,input_dim=1025,activation='tanh'))
    model.add(Dense(50,activation='tanh'))
    model.add(Dense(1, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='Adam') 

########################################################################
########################################################################
def NMF(X_train, n_comp):

    model = sklearn.decomposition.NMF(n_components=n_comp,
                                        init='random',
                                        random_state=1,
                                        max_iter=500,
                                        verbose=True,
                                        tol=5e-5)
    W = model.fit_transform(X_train)
    H = model.components_   

    return W, H

########################################################################
########################################################################
