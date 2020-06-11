import numpy as np 
import library
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os, joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.preprocessing import StandardScaler,Normalizer
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model

# gpu_device = 0
# gpu_frac = 0.5

# os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_device)

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)    
# config = tf.ConfigProto(gpu_options=gpu_options)

X = np.load('data_X.npy')
Y = np.load('data_Y.npy')
print(X.shape)
print(Y.shape)
ind = np.where(Y > 1)
Y = Y[ind]
X = X[ind]
print(X.shape)

Y = Y/10

def transform_X(X):
    norm = joblib.load('normalizer.pkl')
    std = joblib.load('standardscaler.pkl')
    X = norm.transform(X)
    X = std.transform(X)
    return X

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


def fitting(model,X_train,Y_train,epochs=25,batch_size=128):
    
    history = model.fit(X_train,Y_train,epochs=epochs,batch_size=batch_size,verbose=1)
    model.save("model_mir.h5")

    return model


def evaluation(model,X_test,Y_test):

    Y_pred = (model.predict(X_test))[:,0]
    
    #accuracy = ((abs(Y_test-Y_pred[:,0]) < 0.2).sum() / Y_test.shape[0])
    RPA = (abs( 1200*np.log2(Y_pred/Y_test) ) < 50).sum() / Y_test.shape[0]

    print("raw pitch accuracy :",RPA*100)
    
    # plt.plot(np.arange(Y_test.shape[0]),abs(Y_pred[:,0]-Y_test)*10)
    # plt.show()

    return RPA*100
        

if __name__=='__main__':
    
    kf = KFold(n_splits=10, random_state=1, shuffle=True) # Define the split - into 10 folds 
    kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator

    accuracy = []

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        X_train, X_test = preprocessing_X(X_train, X_test)
        model = fitting(library.NN_regressor(),X_train, Y_train,epochs=50,batch_size=128)
        
        # X_test = transform_X(X_test)
        # model = load_model('model_mir.h5')

        accuracy.append(evaluation(model,X_test,Y_test))

        break

    print(accuracy)









