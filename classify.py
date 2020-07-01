import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder,StandardScaler,Normalizer
import config
import time
import joblib
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf 
from library import *
############################################
############################################

X = np.load('data/data_X.npy')
Y = np.load('data/data_Y.npy')
print(Y.shape)

# ind = np.arange(0,X.shape[0],2)
# X = X[ind]
# Y = Y[ind]

ind = np.where(Y <= 0.5)
Y[:] = 1
Y[ind] = 0

Y = (tf.one_hot(Y,2)).numpy()
print(Y.shape)

def preprocessing_X(X_train,X_test):
    
    scaler = Normalizer().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, 'saved_models/normalizer_classifier.pkl') 

    scaler1 = StandardScaler()
    scaler1.fit(X_train)
    joblib.dump(scaler1, 'saved_models/standardscaler_classifier.pkl') 

    X_train = scaler1.transform(X_train)
    X_test = scaler1.transform(X_test)

    return X_train, X_test


def evaluation(model,X_test,Y_test):

    Y_pred = model.predict(X_test)
    scores = model.evaluate(X_test,Y_test)
    
    return scores[1]*100,Y_pred



def main():
    
    kf = KFold(n_splits=5, random_state=1, shuffle=True) # Define the split - into 10 folds 
    kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator

    accuracy = []

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        X_train, X_test = preprocessing_X(X_train, X_test)
        
        model = NN_classifier()
        
        n=100; b=512
        history = model.fit(X_train,Y_train,epochs=n,batch_size=b,verbose=1)
        model.save("saved_models/model_mir_classifier.h5")

        acc, Y_pred = evaluation(model,X_test,Y_test)

        accuracy.append(acc)
        print(accuracy)

        plt.plot(history.history['accuracy'])
        plt.show()

        break


if __name__=='__main__':
    start = time.time()
    main()
    end = time.time()
    # print('TOTAL TIME TAKEN',end-start)