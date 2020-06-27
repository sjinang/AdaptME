import numpy as np 
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os, joblib
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.preprocessing import StandardScaler,Normalizer
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model
from library import *
from config import data_mir
import time
import datetime

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

########################################################################
########################################################################

X = np.load('data_X.npy')
Y = np.load('data_Y.npy')
print(X.shape)

ind = np.where(Y > 1)
Y = Y[ind]
X = X[ind]
print(X.shape)
########################################################################
########################################################################

def preprocessing_X(X_train,X_test):
    
    scaler = Normalizer().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, 'normalizer.pkl') 

    scaler1 = StandardScaler()
    scaler1.fit(X_train)
    joblib.dump(scaler1, 'standardscaler.pkl') 

    X_train = scaler1.transform(X_train)
    X_test = scaler1.transform(X_test)

    return X_train, X_test

def transform_X(X):
    norm = joblib.load('normalizer.pkl')
    std = joblib.load('standardscaler.pkl')
    X = norm.transform(X)
    X = std.transform(X)
    return X

def fitting(model, X_train, Y_train, epochs=25, batch_size=128,const=data_mir.const):
    data_mir.log+=datetime.datetime.now().strftime("%d-%m %H-%M")
    tb = TensorBoard(log_dir='logs/'+(data_mir.log))
    
    history = model.fit(X_train,
                        Y_train/const,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1,
                        # validation_split=0.1,
                        callbacks=[tb])
    model.save("model_mir.h5")

def evaluation(model,X_test,Y_test,const=data_mir.const):
    Y_pred = ((model.predict(X_test))[:,0])*const

    F_test = 440*(np.power(2,(Y_test-69)/12))
    F_pred = 440*(np.power(2,(Y_pred-69)/12))

    RPA = (abs(Y_pred - Y_test) <= 0.5).sum() / (Y_test.shape[0]/100)   
    RFA = ( (abs(F_test-F_pred)/F_test) <= 0.05 ).sum() / (Y_test.shape[0]/100)   
    # print("accuracy :",RPA,RFA)

    return [RPA,RFA], Y_pred
########################################################################
########################################################################

def main():
    
    kf = KFold(n_splits=10, random_state=1, shuffle=True) # Define the split - into 10 folds 
    kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator

    accuracy = []

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        X_train, X_test = preprocessing_X(X_train, X_test)
        
        model = NN_regressor()
        
        n=200; b=512
        data_mir.log+='_'+str(n)+'_'+str(b)+'_'
        fitting(model, X_train, Y_train, epochs=n, batch_size=b)
        
        # X_test = transform_X(X_test)
        # X_train = transform_X(X_train)
        # model = load_model('model_mir.h5')

        ra_test, Y_pred_test = evaluation(model,X_test,Y_test)
        ra_train,Y_pred_train = evaluation(model,X_train,Y_train)
        accuracy.append([ra_test,ra_train])

        break

    print(accuracy)


if __name__=='__main__':
    start = time.time()
    main()
    end = time.time()
    print('TOTAL TIME TAKEN',end-start)







