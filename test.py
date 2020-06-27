import numpy as np 
# import matplotlib.pyplot as plt
# import librosa
# from library import *
# import os, glob
# import librosa.display
# from sklearn.preprocessing import StandardScaler,normalize
# from config import data_mir
# from tensorflow.keras.models import load_model
# import joblib
# from tensorflow.keras.models import model_from_json
# from config import data_mir
import tensorflow as tf

def rpa(y_true,y_pred):
    a = tf.abs(y_true-y_pred) < 0.5
    a = tf.cast(a,dtype=tf.float32)
    return tf.reduce_mean(a)
t = tf.constant(6)
t1 = tf.constant([1.,2.,3.,4.,5.])
t2 = tf.constant([1.6,2.2,2.9,4.4,4.51])
# print(tf.make_ndarray(rpa(t1,t2)))
with tf.Session() as sess:  print(rpa(t1,t2).eval()) 







# path3 = './mir-1k/Wavfile'
# path4 = './mir-1k/PitchLabel'
# def generate_data_single(name,sr=data_mir.SR,Hs=data_mir.Hs,Ws=data_mir.Ws,N_fft=data_mir.N_fft):

#     f_path1 = os.path.join(path3,name+'.wav')
#     f_path2 = os.path.join(path4,name+'.pv')

#     wav, sr = librosa.core.load(f_path1,sr=sr)
#     HOP_len = int(sr * Hs)
#     WIN_len = int(sr * Ws)

#     temp = np.abs(librosa.core.stft(wav,hop_length=HOP_len,win_length=WIN_len,n_fft=N_fft))
#     # e = pow(10,-2)
#     # temp = 20*np.log10(temp+e)
#     temp = librosa.core.amplitude_to_db(temp,ref=np.mean)

#     pitch_vals = np.loadtxt(f_path2)

#     minm = min(temp.shape[1],pitch_vals.shape[0])
    
#     X = (temp.T)[:minm]
#     Y = pitch_vals[:minm]

#     return X,Y

# def transform_X(X):
#     norm = joblib.load('normalizer.pkl')
#     std = joblib.load('standardscaler.pkl')
#     X = norm.transform(X)
#     X = std.transform(X)
#     return X

# def evaluation(model,X_test,Y_test,const=data_mir.const):
#     Y_pred = ((model.predict(X_test))[:,0])*const 
#     RPA = (abs(Y_pred - Y_test) <= 0.5).sum() / (Y_test.shape[0]/100)   

#     return RPA, Y_pred


# f_path = './mir-1k/Wavfile/abjones_1_04.wav'
# p_path = './mir-1k/PitchLabel/abjones_1_01.pv'
# name = 'amy_2_04'
# ###################################################################

# X, Y = generate_data_single(name)

# ind = np.where(Y > 1)
# Y = Y[ind]
# X = X[ind]

# model = load_model('model_mir.h5')

# X = transform_X(X)
# rpa, Y_pred = evaluation(model,X,Y)
# print('****************************************',rpa)

# plt.plot(np.arange(Y.shape[0]),Y,'g')
# plt.plot(np.arange(Y.shape[0]),Y_pred,'y')
# plt.show()










# Hs = data_mir.Hs
# Ws = data_mir.Ws
# N_fft = data_mir.N_fft
# SR = data_mir.SR
# wav, sr = librosa.core.load(f_path,sr=SR)
# print(wav.shape)
# HOP_len = int(sr * Hs)
# WIN_len = int(sr * Ws)

# # print("SR",SR,"HOP",HOP_len,"WIN", WIN_len)
 
# X = np.abs(librosa.core.stft(wav,hop_length=HOP_len,win_length=WIN_len,n_fft=N_fft,window='hann'))
# print(X.shape)

# # plt.figure(1)
# # b = librosa.display.specshow(librosa.core.amplitude_to_db(X,ref=np.max),y_axis='log', x_axis='s',sr=sr,hop_length=HOP_len)
# # plt.title('Power spectrogram')
# # plt.colorbar(format='%+2.0f dB')
# # plt.tight_layout()
# # plt.show()

# Y = np.loadtxt(p_path)
# plt.figure(2)
# plt.plot(np.arange(0,Y.shape[0]*0.02,0.02),Y,'o',markersize=3)
# #plt.yscale('log')
# plt.show()






# f_path = './Orchset/GT/Beethoven-S3-I-ex1.mel'
# pitch_vals = np.loadtxt(f_path)
# Y = pitch_vals[:,1]
#print(X[118,199])
# print(X.shape,Y.shape,sr)

# X1 = StandardScaler()
# X1.fit(X.T)
# X = (X1.transform(X.T)).T
#X = normalize(X.T).T

# fig = plt.figure(1)
# ax = fig.add_subplot(111)
# e = pow(10,-2)
# pc = ax.pcolormesh(np.arange(X.shape[1]),np.arange(X.shape[0])*(sr/N_fft),20*np.log10(X+e),snap=True,cmap='magma')
# print((X.shape[0]-1)*(sr/N_fft))
# fig.colorbar(pc)
# plt.ylim(0,(X.shape[0]-1)*(sr/N_fft))
# #plt.yscale('log')
# plt.show()

# plt.figure(2)
# b= librosa.display.specshow(librosa.amplitude_to_db(X,ref=np.max),y_axis='hz',sr=SR,hop_length=HOP_len)
# plt.title('Power spectrogram')
# plt.colorbar(format='%+2.0f dB')
# plt.tight_layout()
# plt.show()
            


    


    
    

