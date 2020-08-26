import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential, load_model
from sklearn import metrics
import sklearn.decomposition
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.model_selection import KFold
import librosa, librosa.display
import pickle, dill, joblib, os, time, datetime
from tensorflow.keras.regularizers import l1,l2
from tensorflow.keras import Input,Model
from config import data_mir



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
                        verbose=1
                        # , validation_split=0.05
                        # , callbacks=[tb]
                        )
    model.save("saved_models/model_mir.h5")


def evaluation(model,X_test,Y_test,const=data_mir.const):
    Y_pred = ((model.predict(X_test))[:,0])*const

    M_test = (12*np.log2(Y_test/440)) + 69
    M_pred = (12*np.log2(Y_pred/440)) + 69

    RPA = (abs(M_pred - M_test) <= 0.5).sum() / (M_test.shape[0]/100)   
    RFA = ( (abs(Y_test-Y_pred)/Y_test) <= 0.05 ).sum() / (Y_test.shape[0]/100)   
    # print("accuracy :",RPA,RFA)

    return [RPA,RFA], Y_pred