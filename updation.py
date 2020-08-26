from library import *
from sklearn.model_selection import train_test_split



path3 = './mir-1k/Wavfile'
path4 = './mir-1k/PitchLabel'


def generate_data_single(name,sr=data_mir.SR,Hs=data_mir.Hs,Ws=data_mir.Ws,N_fft=data_mir.N_fft):

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
    
    return X,Y[:,1]


def transform_X(X):
    norm = joblib.load('saved_models/normalizer.pkl')
    std = joblib.load('saved_models/standardscaler.pkl')
    X = norm.transform(X)
    X = std.transform(X)
    return X


def addn_fitting(model,X_train,Y_train,epochs=2,batch_size=128,const=data_mir.const):
    history = model.fit(X_train,
                        Y_train/const,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1)
    
    model.save("saved_models/model_mir_addn.h5")
    return model

   
def evaluation(model,X_test,Y_test,const=data_mir.const):
    Y_pred = ((model.predict(X_test))[:,0])*const

    M_test = (12*np.log2(Y_test/440)) +69
    M_pred = (12*np.log2(Y_pred/440)) +69

    RPA = (abs(M_pred - M_test) <= 0.5).sum() / (Y_test.shape[0]/100)   
    # print("raw pitch accuracy :",RPA)

    return RPA, Y_pred


Name = np.loadtxt('./mir-1k/Only_filenames.txt',dtype='str')[:10]
old_RPAs = []
new_RPAs = []
data=[]

for name in Name:

    X, Y = generate_data_single(name,Ws=data_mir.Ws)
    # X = np.load('data/data_X.npy')
    # Y = np.load('data/data_Y.npy')
    # print(X.shape,Y.shape)
    ind = np.where(Y > 1)
    Y = Y[ind]
    X = X[ind]
    # print(X.shape)
    model = load_model('saved_models/model_mir.h5', custom_objects={'RPA': RPA})

    X = transform_X(X)
    rpa, Y_pred = evaluation(model,X,Y)
    
    M = (12*np.log2(Y/440)) +69
    M_pred = (12*np.log2(Y_pred/440)) +69

    dev = abs(M_pred-M)
    ind = np.argsort(dev)[::-1][:64]

    X_train = X[ind]
    Y_train = Y[ind] 

    rpa_1, Y_pred_1 = evaluation(model,X_train,Y_train)
    model = addn_fitting(model,X_train,Y_train,10)

    rpa_2, Y_pred_2 = evaluation(model,X_train,Y_train)

    # effect of new trained model on original dataset
    rpa_new, Y_pred_new = evaluation(model,X,Y)

    old_RPAs.append(rpa)
    new_RPAs.append(rpa_new)

    data.append([rpa, rpa_new,rpa_1,rpa_2])

old_RPAs = np.array(old_RPAs)
new_RPAs = np.array(new_RPAs)

for d in data:
    print(d[0],d[1])


# print( ((old_RPAs - new_RPAs) < 0).sum() * 2 )