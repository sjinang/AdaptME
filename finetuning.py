from library import *
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from sklearn.metrics import accuracy_score
def generate_data_single(name,SR=data_mir.SR,Hs=data_mir.Hs,Ws=data_mir.Ws,N_fft=data_mir.N_fft):

    f_path1 = os.path.join('mirex05',name+'.wav')
    f_path2 = os.path.join('mirex05',name+'REF.txt')
    print(f_path1,f_path2)
    wav, sr = librosa.core.load(f_path1,sr=SR)
    HOP_len = int(sr * Hs)
    WIN_len = int(sr * Ws)

    temp = np.abs(librosa.core.stft(wav,hop_length=HOP_len,win_length=WIN_len,n_fft=N_fft))
    temp = librosa.core.amplitude_to_db(temp,ref=np.max)
    # e = pow(10,-2)
    # temp = 20*np.log10(temp+e)

    pitch_vals = np.loadtxt(f_path2)

    minm = min(temp.shape[1],pitch_vals.shape[0])
    
    X = (temp.T)[:minm]
    Y = pitch_vals[:minm]

    return X,Y,temp

X,Y,temp = generate_data_single('train05')
print(X.shape)
ind = np.where(Y[:,1]>1)
X = X[ind]
Y= Y[ind]
print(X.shape)
X = transform_X(X)

model = load_model('saved_models/model_class.h5',) #, custom_objects={'RPA': RPA}

# ind = [54,567,983,190,764]
# X_train = X[ind]
# Y_train = Y[ind,1]

# model.fit(X_train,Y_train,epochs=1000)

Y_pred = np.argmax(model.predict(X),-1)
Y_true = freq2midi(Y[:,1]).round() - 36




print('RPA :',accuracy_score(Y_true,Y_pred))

plt.plot(Y[:,0],Y_true,'go',markersize=5)
plt.plot(Y[:,0],Y_pred,'yo',markersize=2)
plt.show()

# plt.figure(1)
# plt.plot(Y[:,0],Y[:,1],'o',markersize=0.5)

# b= librosa.display.specshow(temp,y_axis='hz', x_axis='s',sr=16000,hop_length=160)
# plt.title('Power spectrogram')
# plt.colorbar(format='%+2.0f dB')
# plt.tight_layout()
# plt.show()