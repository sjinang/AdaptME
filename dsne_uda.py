import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from library import *

def seed_all(seed=2020):
    np.random.seed(seed)
    tf.random.set_seed(seed)
seed_all()
#########################################################
#########################################################

# import numpy as np
# a = tf.constant([[1,2],[3,6],[1,0]])
# b = tf.constant([15,8])
# temp = np.tile(np.square(np.linalg.norm(a,axis=1)),(b.shape[0],1)) + np.tile(np.square(np.linalg.norm(b,axis=1)),(a.shape[0],1)).T - (2*(np.matmul(b,a.T)))
# print(tf.norm(a-b,1,axis=1))
def generate_data_single(name,SR=data_mir.SR,Hs=data_mir.Hs,Ws=data_mir.Ws,N_fft=data_mir.N_fft):
    f_path1 = os.path.join('mirex05',name+'.wav')
    f_path2 = os.path.join('mirex05',name+'REF.txt')
    
    wav, sr = librosa.core.load(f_path1,sr=SR)
    HOP_len = int(sr * Hs)
    WIN_len = int(sr * Ws)

    temp = np.abs(librosa.core.stft(wav,hop_length=HOP_len,win_length=WIN_len,n_fft=N_fft))
    temp = librosa.core.amplitude_to_db(temp,ref=np.mean)
    pitch_vals = np.loadtxt(f_path2)

    minm = min(temp.shape[1],pitch_vals.shape[0])
    X = (temp.T)[:minm]
    Y = pitch_vals[:minm,1]

    return X,Y

#########################################################
#########################################################

target_X,target_Y = generate_data_single('train06')
target_X = target_X[target_Y>1]
target_Y = target_Y[target_Y>1]
target_X = transform_X(target_X[:,:512])

ind = np.arange(target_X.shape[0])
# ind = np.array([2093,2127,2052,647,714,741,641,548,558,756,31,37,143,49,1241,301])
np.random.shuffle(ind)
x_tar = tf.convert_to_tensor(np.take(target_X,ind[:16],0))
y_tar = tf.convert_to_tensor(np.take(target_Y,ind[:16],0))

model = NN_regressor_dsne(512)
model.load_weights('saved_models/model_mir_for_dsne_weights.h5')

# obj = np.empty(n_class,dtype=object)
# for c in range(n_class):
#     obj[i] = np.argwhere(np.round(freq2midi(y_src))==c)[0]

# tar = tf.data.Dataset.from_tensor_slices((x_tar, y_tar))
# tar = tar.shuffle(y_tar.shape[0]+1, reshuffle_each_iteration=True)

def hausdorffian_distance(xt,y,feat_src,y_src):
    y_src = y_src.numpy()
    y = y.numpy()
    ind_y = np.argwhere(np.round(freq2midi(y_src))==np.round(freq2midi(y)))[0]
    all_ind = np.arange(feat_src.shape[0])
    ind_not_y = np.setxor1d(all_ind, ind_y)

    same_class_max = tf.reduce_max(tf.norm(tf.gather(feat_src,ind_y)-xt,2,axis=1))
    diff_class_min = tf.reduce_min(tf.norm(tf.gather(feat_src,ind_not_y)-xt,2,axis=1))

    return same_class_max-diff_class_min

pred = model()




# ind_1 = np.arange(100)
# opt = tf.keras.optimizers.Adam(0.0005)
# loss_mse = tf.keras.losses.MeanAbsoluteError()

# epochs = 25; alpha=0.5; beta=1.0
# acc = []
# for epoch in range(epochs):
#     # tar = tar.shuffle(len_tar+1)
#     count=0
#     np.random.shuffle(ind_1)
#     for tar_x,tar_y in tar:
#         for src_ind in ind_1[:10]:
#             x_src = tf.convert_to_tensor(np.load('data/dsne/src/src_X_t_{}.npy'.format(src_ind)))
#             y_src = tf.convert_to_tensor(np.load('data/dsne/src/src_Y_{}.npy'.format(src_ind)))

#             with tf.GradientTape() as tape:
#                 out_src = model(x_src,training=True)
#                 out_tar = model(x_tar,training=True)
#                 out_tar_sample = model(tf.reshape(tar_x,[1,tar_x.shape[0]]),training=True)
                
#                 loss_mse_tot = loss_mse(y_src,out_src[0]) + beta*loss_mse(y_tar,out_tar[0])
#                 loss_value= (
#                             (1-alpha)*loss_mse_tot
#                             + alpha*hausdorffian_distance(out_tar_sample[1],tar_y,out_src[1],y_src)
#                             )   

#             grads = tape.gradient(loss_value,model.trainable_weights)
#             opt.apply_gradients(zip(grads, model.trainable_weights))

#             print('epoch',epoch,'target count',count,'source',src_ind)
        
#         count+=1
    
#     rpa,_ = evaluation_dsne(model,target_X,target_Y)
#     acc.append(rpa)
#     print('############################')
#     print('epoch',epoch,'target_RPA',rpa)
#     print('############################')

# # model.save_weights('saved_models/dsne/model_dsne_weights.h5')
# acc.reverse()
# acc.append(0.5)
# acc.reverse()
# acc = np.array(acc)
# print(np.max(acc),np.argmax(acc))
# plt.plot(np.arange(epochs+1),acc,'o-')
# plt.show()