import numpy as np 
import matplotlib.pyplot as plt
import librosa
import os, glob
import librosa.display
from config import data_mir

# path1 = './Orchset/audio/mono'
# path2 = './Orchset/GT'
path3 = './mir-1k/Wavfile'
path4 = './mir-1k/PitchLabel'

def generate_data_single(name,SR=data_mir.SR,Hs=data_mir.Hs,Ws=data_mir.Ws,N_fft=data_mir.N_fft):

    f_path1 = os.path.join(path3,name+'.wav')
    f_path2 = os.path.join(path4,name+'.pv')

    wav, sr = librosa.core.load(f_path1,sr=SR)
    HOP_len = int(sr * Hs)
    WIN_len = int(sr * Ws)

    temp = np.abs(librosa.core.stft(wav,hop_length=HOP_len,win_length=WIN_len,n_fft=N_fft))
    e = pow(10,-2)
    temp = 20*np.log10(temp+e)

    pitch_vals = np.loadtxt(f_path2)

    minm = min(temp.shape[1],pitch_vals.shape[0])
    
    X = (temp.T)[:minm]
    Y = pitch_vals[:minm]

    return X,Y

def generate_data_or(Hs=0.01, Ws=0.05, N_fft=2048):

    # if os.path.exists('data_X.txt') and os.path.exists('data_Y.txt') and os.path.exists('data_SR.txt'):
    #     return

    X = np.array([])
    Y = np.array([])
    SR = 7350
    interval = []

    count = 0
    print("")
    for f_path in glob.glob(os.path.join(path1, '*.wav')):
        
        wav, sr = librosa.core.load(f_path,duration=9,sr=SR)
        HOP_len = int(sr * Hs)
        WIN_len = int(sr * Ws)

        temp = np.abs(librosa.core.stft(wav,hop_length=HOP_len,win_length=WIN_len,n_fft=N_fft))
        e = pow(10,-2)
        temp = 20*np.log10(temp+e)

        interval.append(temp.shape[1])

        if count==0:
            X = temp.T
        else:
            X = np.append(X,temp.T,axis=0)
        count+=1
        print(" * X_{} done...".format(count),end="\r")

    count=0
    for f_path in glob.glob(os.path.join(path2, '*.mel')):

        pitch_vals = np.loadtxt(f_path)
        pitch_vals = pitch_vals[:interval[count],1]
        #print(interval[count], len(pitch_vals))
        count+=1
        print(" * Y_{} done...".format(count),end="\r")

        Y = np.append(Y,pitch_vals)

    
    np.save('data_X.npy',X)
    np.save('data_Y.npy',Y)
    np.save('data_SR.npy',np.array([SR]))
    
    return X, Y, SR


def generate_data_mir(SR=data_mir.SR,Hs=data_mir.Hs, Ws=data_mir.Ws, N_fft=data_mir.N_fft):
    
    X = np.array([])
    Y = np.array([])
    SR = SR
    count = 0
    print(SR,Hs,Ws,N_fft)
    for f_path1,f_path2 in zip(sorted(glob.glob(os.path.join(path3, '*.wav'))), sorted(glob.glob(os.path.join(path4, '*.pv')))):
    
        wav, sr = librosa.core.load(f_path1,sr=SR)
        HOP_len = int(sr * Hs)
        WIN_len = int(sr * Ws)
        temp = np.abs(librosa.core.stft(wav,hop_length=HOP_len,win_length=WIN_len,n_fft=N_fft))
        
        # e = pow(10,-2)
        # temp = 20*np.log10(temp+e)
        temp = librosa.core.amplitude_to_db(temp,ref=np.mean)

        pitch_vals = np.loadtxt(f_path2)

        minm = min(temp.shape[1],pitch_vals.shape[0])

        if count==0:
            X = (temp.T)[:minm]
        else:
            X = np.append(X,(temp.T)[:minm],axis=0)
        
        Y = np.append(Y,pitch_vals[:minm])
        
        count+=1
        print(" * X & Y {} done...".format(count),end="\r")

    
    np.save('data_X.npy',X)
    np.save('data_Y.npy',Y)
    np.save('data_SR.npy',np.array([SR]))
    
    print("X.shape :", X.shape, "Y.shape :", Y.shape)



def generate_data_mir1(SR=data_mir.SR,Hs=data_mir.Hs, Ws=data_mir.Ws, N_fft=data_mir.N_fft):
    
    X = []
    Y = []
    x = []
    y = []
    SR = SR
    count = 0
    print(SR,Hs,Ws,N_fft)
    for f_path1,f_path2 in zip(sorted(glob.glob(os.path.join(path3, '*.wav'))), sorted(glob.glob(os.path.join(path4, '*.pv')))):
    
        wav, sr = librosa.core.load(f_path1,sr=SR)
        HOP_len = int(sr * Hs)
        WIN_len = int(sr * Ws)
        temp = np.abs(librosa.core.stft(wav,hop_length=HOP_len,win_length=WIN_len,n_fft=N_fft))
        
        # e = pow(10,-2)
        # temp = 20*np.log10(temp+e)
        temp = librosa.core.amplitude_to_db(temp,ref=np.mean)

        pitch_vals = np.loadtxt(f_path2)

        minm = min(temp.shape[1],pitch_vals.shape[0])

        if count%10==0:
            X.extend(x)
            Y.extend(y)
            x = ((temp.T)[:minm]).tolist()
            y = (pitch_vals[:minm]).tolist()
        else:
            x.extend(((temp.T)[:minm]).tolist())
            y.extend((pitch_vals[:minm]).tolist())
        
        count+=1
        print(" * X & Y {} done...".format(count),end="\r")

    X.extend(x)
    Y.extend(y)
    
    np.save('data_X.npy',np.array(X))
    np.save('data_Y.npy',np.array(Y))
    np.save('data_SR.npy',np.array([SR]))
    
    print("X.shape :", X.shape, "Y.shape :", Y.shape)






# plt.figure(1)
# b= librosa.display.specshow(librosa.amplitude_to_db(a,ref=np.max),y_axis='log', x_axis='s',sr=sr,hop_length=HOP_len)
# plt.title('Power spectrogram')
# plt.colorbar(format='%+2.0f dB')
# plt.tight_layout()
# plt.show()
            
generate_data_mir()
