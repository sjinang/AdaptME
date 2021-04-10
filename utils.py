import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Sequential, load_model
from sklearn import metrics
import sklearn.decomposition
from sklearn.preprocessing import StandardScaler,Normalizer, MinMaxScaler
from sklearn.model_selection import KFold
import librosa, librosa.display, gc
import pickle, dill, joblib, os, time, datetime
from tensorflow.keras.regularizers import l1,l2
from tensorflow.keras import Model
from config import data_mir
from sklearn.metrics import accuracy_score

def freq2midi(f):
    return 69 + 12*np.log2(f/440)


def midi2freq(m):
    return 2**((m - 69)/ 12) * 440


def RPA(y_true,y_pred):
    y_true = ((12*tf.math.log(y_true/440)) / tf.math.log( tf.constant(2,dtype=tf.float32) )) + 69
    y_pred = ((12*tf.math.log(y_pred/440)) / tf.math.log( tf.constant(2,dtype=tf.float32) )) + 69
    # a = tf.abs(y_true-y_pred) < 0.5
    # a = tf.cast(a,dtype=tf.float32)
    # return 100*tf.reduce_mean(a)
    y_true = tf.math.round(y_true)
    y_pred = tf.math.round(y_pred)
    return tf.reduce_mean(tf.cast(y_true==y_pred,dtype=tf.float32))


def preprocessing_X(X_train,X_test):
    
    scaler = Normalizer().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, 'saved_models/normalizer.pkl') 

    scaler1 = StandardScaler()
    scaler1.fit(X_train)
    joblib.dump(scaler1, 'saved_models/standardscaler.pkl') 

    X_train = scaler1.transform(X_train)
    X_test = scaler1.transform(X_test)

    return X_train, X_test


def transform_X(X):
    norm = joblib.load('saved_models/normalizer.pkl')
    std = joblib.load('saved_models/standardscaler.pkl')
    X = norm.transform(X)
    X = std.transform(X)
    return X


def fitting(model, X_train, Y_train, epochs=25, batch_size=512,const=data_mir.const):
    # data_mir.log+=datetime.datetime.now().strftime("%d-%m %H-%M")
    # tb = tf.keras.callbacks.TensorBoard(log_dir='logs/'+(data_mir.log))
    
    history = model.fit(X_train,
                        Y_train/const,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=2
                        , validation_split=0.1
                        # , callbacks=[tb]
                        )
    model.save("saved_models/model_mir1k.h5")


def evaluation(model,X_test,Y_test,const=data_mir.const):
    Y_pred = (model(X_test,training=False)[:,0])*const

    M_test = freq2midi(Y_test)
    M_pred = freq2midi(Y_pred)

    M_pred = M_pred.round()
    M_true = M_test.round()
    return accuracy_score(M_true,M_pred), Y_pred

def evaluation_dsne(model,X_test,Y_test,const=data_mir.const):
    Y_pred = (model(X_test,training=False)[0][:,0])*const

    M_test = freq2midi(Y_test)
    M_pred = freq2midi(Y_pred)

    M_pred = M_pred.round()
    M_true = M_test.round()
    return accuracy_score(M_true,M_pred), Y_pred