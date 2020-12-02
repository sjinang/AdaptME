import numpy as np 
import matplotlib.pyplot as plt
import librosa
import os, glob
import librosa.display
from config import data_mir
import gc
from utils import freq2midi
# path1 = './Orchset/audio/mono'
# path2 = './Orchset/GT'
# path3 = 'mir-1k/Wavfile'
# path4 = 'mir-1k/PitchLabel'
path3 = 'mirex05/'
path4 = 'mirex05/'

def generate_data_single(name,SR=data_mir.SR,Hs=data_mir.Hs,Ws=data_mir.Ws,N_fft=data_mir.N_fft):

    f_path1 = os.path.join(path3,name+'.wav')
    # f_path2 = os.path.join(path4,name+'.pv')
    f_path2 = os.path.join(path4,name+'REF.txt')

    wav, sr = librosa.core.load(f_path1,sr=SR)
    HOP_len = int(sr * Hs)
    WIN_len = int(sr * Ws)

    temp = np.abs(librosa.core.stft(wav,hop_length=HOP_len,win_length=WIN_len,n_fft=N_fft))
    temp = librosa.core.amplitude_to_db(temp,ref=np.mean)
    # e = pow(10,-2)
    # temp = 20*np.log10(temp+e)

    pitch_vals = np.loadtxt(f_path2)

    minm = min(temp.shape[1],pitch_vals.shape[0])
    
    X = (temp.T)[:minm]
    Y = pitch_vals[:minm,1]

    return X,Y,temp



def generate_data_mir(SR=data_mir.SR,Hs=data_mir.Hs, Ws=data_mir.Ws, N_fft=data_mir.N_fft):
    mina = []
    X = np.array([])
    Y = np.array([])
    count = 0
    print(SR,Hs,Ws,N_fft)
    for f_path1,f_path2 in zip(sorted(glob.glob(os.path.join(path3, '*.wav'))), sorted(glob.glob(os.path.join(path4, '*.pv')))):
    
        wav, sr = librosa.core.load(f_path1,sr=SR)
        HOP_len = int(sr * Hs)
        WIN_len = int(sr * Ws)
        temp = np.abs(librosa.core.stft(wav,hop_length=HOP_len,win_length=WIN_len,n_fft=N_fft))
        
        temp = librosa.core.amplitude_to_db(temp,ref=np.mean)

        pitch_vals = np.loadtxt(f_path2)
    
        minm = min(temp.shape[1],pitch_vals.shape[0])
        mina.append(minm)
        # if count==0:
        #     X = (temp.T)[:minm]
        # else:
        #     X = np.append(X,(temp.T)[:minm],axis=0)
        
        # Y = np.append(Y,pitch_vals[:minm][:,1])
        
        count+=1
        # print(" * X & Y {} done...".format(count),end="\r")
        print(count,f_path1)
    np.savetxt('data/framespersong.txt',np.array(mina))
    # print("X.shape :", X.shape, "Y.shape :", Y.shape)
    
    # np.save('data/8k/data_X.npy',X)
    # np.save('data/8k/data_Y.npy',Y)
    # np.save('data/8k/data_SR.npy',np.array([SR]))
    
    
def generate_data_mir_shift(SR=data_mir.SR,Hs=data_mir.Hs, Ws=data_mir.Ws, N_fft=data_mir.N_fft):
    mina = []
    X = np.array([])
    Y = np.array([])
    count = 0
    shift = 6
    print(SR,Hs,Ws,N_fft)
    
    for f_path1,f_path2 in zip(sorted(glob.glob(os.path.join(path3, '*.wav'))), sorted(glob.glob(os.path.join(path4, '*.pv')))):
        # print(f_path1)
        wav, sr = librosa.core.load(f_path1,sr=SR)
        HOP_len = int(sr * Hs)
        WIN_len = int(sr * Ws)
        wav = librosa.effects.pitch_shift(wav, sr, n_steps=shift)
        
        temp = np.abs(librosa.core.stft(wav,hop_length=HOP_len,win_length=WIN_len,n_fft=N_fft))
        
        temp = librosa.core.amplitude_to_db(temp,ref=np.mean)

        pitch_vals = np.loadtxt(f_path2)[:,1]
    
        minm = min(temp.shape[1],pitch_vals.shape[0])
        mina.append(minm)
        if count==0:
            X = (temp.T)[:minm]
        else:
            X = np.append(X,(temp.T)[:minm],axis=0)
        
        Y = np.append(Y,pitch_vals[:minm])
        
        count+=1
        # print(" * X & Y {} done...".format(count),end="\r")
        print(count,f_path1)
    
    print("X.shape :", X.shape, "Y.shape :", Y.shape)
    
    np.save('data/data_X_6s.npy',X)
    np.save('data/data_Y_6s.npy',Y)


# generate_data_mir_shift()


# X = np.load('data/data_X_1.npy')
# Y = np.load('data/data_Y_1.npy')
# print(X.shape)
# # print(Y[:10])
# Y1 = freq2midi(Y).round()
# print(Y1.shape[0])
# for l in range(100):
#     ind = np.where(Y1==l)[0]
#     temp = ind.shape[0]
#     ext = int(0.6*ind.shape[0])
#     if ext==0:
#         continue
#     np.random.shuffle(ind)
#     ind = ind[:ext]
#     X = np.delete(X,ind,0)
#     Y = np.delete(Y,ind,0)
#     Y1 = np.delete(Y1,ind,0)
#     print(l,temp,ext,Y.shape[0])

# print(X.shape,Y.shape)
# np.save('data/data_X_10.npy',X)
# np.save('data/data_Y_10.npy',Y)

# plt.hist(Y,np.arange(35,90),histtype='step')
# plt.show()

############################################
from library import *


names = np.loadtxt('mirex05/filenames.txt',dtype=str)
# model = load_model('saved_models/model_mir_for_dsne.h5',custom_objects={'RPA':RPA})
# print(names[:10])
model = NN_regressor_dsne(512)
model.load_weights('saved_models/model_mir_for_dsne_weights.h5')
# names=['train02']
for name in names:
    X,Y,temp = generate_data_single(name)
    X = X[Y>1]
    Y = Y[Y>1]

    # ind = np.array([2093,2127,2052,647,714,741,641,548,558,756,31,37,143,49,1241,301])

    X = transform_X(X[:,:512])
    rpa,pred = evaluation_dsne(model,X,Y)
    print(name,rpa)

    # print(name,rpa)
    # plt.plot(np.arange(pred.shape[0]),pred,'o-',markersize=4)
    # plt.plot(np.arange(pred.shape[0]),Y,'yo',markersize=4)
    # plt.plot(ind,np.take(pred,ind,0),'ro',markersize=10)
    # plt.plot(ind,np.take(Y,ind,0),'ko',markersize=10)
    # plt.xlabel('N with hopsize 10ms')
    # plt.ylabel('frequency')
    # plt.show()


# wav, sr = librosa.core.load('data/shifted_2.wav',sr=16000)
# HOP_len = int(16000 * 0.01)
# WIN_len = int(sr * 12*0.01)

# temp = np.abs(librosa.core.stft(wav,hop_length=HOP_len,win_length=WIN_len,n_fft=2048))
# temp = librosa.core.amplitude_to_db(temp,ref=np.max)

# plt.figure(2)
# b= librosa.display.specshow(temp,y_axis='log', x_axis='s',sr=16000,hop_length=160)
# plt.title('Power spectrogram')
# plt.colorbar(format='%+2.0f dB')
# plt.tight_layout()





# plt.show()

# X,Y,temp = generate_data_single('abjones_1_01',N_fft=2048,Ws=25*0.01)
# plt.figure(2)
# b= librosa.display.specshow(temp,y_axis='hz', x_axis='s',sr=8000,hop_length=80)
# plt.title('Power spectrogram')
# plt.colorbar(format='%+2.0f dB')
# plt.tight_layout()
# plt.show()
# generate_data_mir()
# a = np.array(['abjones','amy','Ani','annar','ariel','bobon','bug','davidson'])


# plt.imshow(X)
# plt.show()
# plt.imshow(temp)
# plt.show()