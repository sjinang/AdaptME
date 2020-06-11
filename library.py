import numpy as np
import sklearn.decomposition
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

def NMF(X_train, n_comp):

    model = sklearn.decomposition.NMF(n_components=n_comp,init='random',random_state=1,max_iter=500,verbose=True,tol=5e-5)
    W = model.fit_transform(X_train)
    H = model.components_   

    return W, H

def NN_regressor():
    model1 = Sequential()
    model1.add(Dense(100, input_dim=1025, activation='tanh')) # ^softsign,tanh
    model1.add(Dense(100, activation='tanh'))
    model1.add(Dense(1, activation='relu'))
    model1.compile(loss='mean_absolute_error', optimizer='Adam')

    return model1 

