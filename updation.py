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
from config import data_mir


path3 = './mir-1k/Wavfile'
path4 = './mir-1k/PitchLabel'


def generate_data_single(name,sr=16000,Hs=0.02,Ws=0.04,N_fft=2048):

    f_path1 = os.path.join(path3,name+'.wav')
    f_path2 = os.path.join(path4,name+'.pv')

    wav, sr = librosa.core.load(f_path1,sr=sr)
    HOP_len = int(sr * Hs)
    WIN_len = int(sr * Ws)

    temp = np.abs(librosa.core.stft(wav,hop_length=HOP_len,win_length=WIN_len,n_fft=N_fft))
    # e = pow(10,-2)
    # temp = 20*np.log10(temp+e)
    temp = librosa.core.amplitude_to_db(temp,ref=np.mean)

    pitch_vals = np.loadtxt(f_path2)

    minm = min(temp.shape[1],pitch_vals.shape[0])
    
    X = (temp.T)[:minm]
    Y = pitch_vals[:minm]

    return X,Y


def transform_X(X):
    norm = joblib.load('normalizer.pkl')
    std = joblib.load('standardscaler.pkl')
    X = norm.transform(X)
    X = std.transform(X)
    return X


def addn_fitting(model,X_train,Y_train,epochs=2,batch_size=128,const=data_mir.const):
    history = model.fit(X_train,
                        Y_train/const,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1)
    
    model.save("model_mir_addn.h5")
    return model

   
def evaluation(model,X_test,Y_test,const=data_mir.const):
    Y_pred = ((model.predict(X_test))[:,0])*const

    RPA = (abs(Y_pred - Y_test) <= 0.5).sum() / (Y_test.shape[0]/100)   
    print("raw pitch accuracy :",RPA)

    return RPA, Y_pred


Name = np.loadtxt('./mir-1k/Only_filenames.txt',dtype='str')[:1]
old_RPAs = []
new_RPAs = []
for name in Name:

    X, Y = generate_data_single(name,Ws=0.12)
    # X = np.load('data_X.npy')
    # Y = np.load('data_Y.npy')

    ind = np.where(Y > 1)
    Y = Y[ind]
    X = X[ind]

    #print('*** X.shape :',X.shape)

    model = load_model('model_mir.h5')

    X = transform_X(X)
    rpa, Y_pred = evaluation(model,X,Y)

    dev = abs((Y_pred-Y)/(Y_pred+Y))
    ind = np.argsort(dev)[::-1][:128]

    X_train = X[ind]
    Y_train = Y[ind] 

    rpa_1, Y_pred_1 = evaluation(model,X_train,Y_train)
    model = addn_fitting(model,X_train,Y_train/10)

    rpa_2, Y_pred_2 = evaluation(model,X_train,Y_train)

    # effect of new trained model on original dataset
    rpa_new, Y_pred_new = evaluation(model,X,Y)
    # print("*** FOR THE WHOLE SONG > OLD RPA :",rpa,"; NEW RPA :",rpa_new)
    
    # plt.plot(abs((Y_pred_1-Y_train)/(Y_pred_1+Y_train)),'bo-')
    # plt.plot(abs((Y_pred_2-Y_train)/(Y_pred_2+Y_train)),'ro-')
    # plt.show()

    old_RPAs.append(rpa)
    new_RPAs.append(rpa_new)

old_RPAs = np.array(old_RPAs)
new_RPAs = np.array(new_RPAs)

for rpa,rpa_new in zip(old_RPAs,new_RPAs):
    print(rpa,' ',rpa_new)


print( ((old_RPAs - new_RPAs) < 0).sum() * 2 )