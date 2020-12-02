from import_libs import *


def generate_data_single(name='daisy1',SR=data_mir.SR,Hs=data_mir.Hs,Ws=data_mir.Ws,N_fft=data_mir.N_fft):

    f_path1 = 'mir-1k/Wavfile/abjones_1_01'+'.wav'
    f_path2 = 'mir-1k/PitchLabel/abjones_1_01'+'.pv'

    wav, sr = librosa.core.load(f_path1,sr=SR)
    HOP_len = int(sr * 10 * 1e-3)
    WIN_len = HOP_len*25

    temp = np.abs(librosa.core.stft(wav,hop_length=HOP_len,win_length=WIN_len,n_fft=4096))
    
    data = np.loadtxt(f_path2)

    plt.figure(1)
    plt.plot(data[:,0],data[:,1],'go',markersize=0.1)
    b = librosa.display.specshow(librosa.core.amplitude_to_db(temp,ref=np.max),y_axis='hz', x_axis='s',sr=sr,hop_length=HOP_len)
    plt.title('Power spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

    # Y = np.loadtxt(p_path)
    # plt.figure(2)
    # plt.plot(np.arange(0,Y.shape[0]*0.02,0.02),Y,'o',markersize=3)
    # #plt.yscale('log')
    # plt.show()

generate_data_single()














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
            


    


    
    

