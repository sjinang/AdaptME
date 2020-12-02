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
# ind = np.array([2093,2127,2052,647,714,741,641,548,558,756,31,37,143,49,1241,301])
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
n_tar = 16

target_X,target_Y = generate_data_single('train06')
target_X = target_X[target_Y>1]
target_Y = target_Y[target_Y>1]
target_X = transform_X(target_X[:,:512])

model = NN_regressor_dsne(512)
model.load_weights('saved_models/model_mir_for_dsne_weights.h5')

orig_rpa,_ = evaluation_dsne(model,target_X,target_Y)

# ind = np.arange(target_X.shape[0])
# np.random.shuffle(ind)
# x_tar = tf.convert_to_tensor(np.take(target_X,ind[:n_tar],0))
# y_tar = tf.convert_to_tensor(np.take(target_Y,ind[:n_tar],0))
###################################################################
# pred_X = model(target_X,training=False)[0][:,0]
# error = abs(pred_X-target_Y)
# ind = np.argsort(error)[-n_tar:]
# x_tar = tf.convert_to_tensor(np.take(target_X,ind,0))
# y_tar = tf.convert_to_tensor(np.take(target_Y,ind,0))
###################################################################
pred_X = model(target_X,training=False)[0][:,0]
th = [200,300,400]
target_X_1 = target_X[target_Y < th[0]]
target_X_2 = target_X[(th[0] <= target_Y) * (target_Y < th[1])]
target_X_3 = target_X[(th[1] <= target_Y) * (target_Y < th[2])]
target_X_4 = target_X[th[2] <= target_Y]

target_Y_1 = target_Y[target_Y < th[0]]
target_Y_2 = target_Y[(th[0] <= target_Y) * (target_Y < th[1])]
target_Y_3 = target_Y[(th[1] <= target_Y) * (target_Y < th[2])]
target_Y_4 = target_Y[th[2] <= target_Y]

pred_X_1 = pred_X[target_Y < th[0]]
pred_X_2 = pred_X[(th[0] <= target_Y) * (target_Y < th[1])]
pred_X_3 = pred_X[(th[1] <= target_Y) * (target_Y < th[2])]
pred_X_4 = pred_X[th[2] <= target_Y]

ind_1 = np.argsort(abs((pred_X_1-target_Y_1)))[-n_tar//4:]
ind_2 = np.argsort(abs((pred_X_2-target_Y_2)))[-n_tar//4:]
ind_3 = np.argsort(abs((pred_X_3-target_Y_3)))[-n_tar//4:]
ind_4 = np.argsort(abs((pred_X_4-target_Y_4)))[-n_tar//4:]

x_tar_1 = tf.convert_to_tensor(np.take(target_X_1,ind_1,0))
y_tar_1 = tf.convert_to_tensor(np.take(target_Y_1,ind_1,0))
x_tar_2 = tf.convert_to_tensor(np.take(target_X_2,ind_2,0))
y_tar_2 = tf.convert_to_tensor(np.take(target_Y_2,ind_2,0))
x_tar_3 = tf.convert_to_tensor(np.take(target_X_3,ind_3,0))
y_tar_3 = tf.convert_to_tensor(np.take(target_Y_3,ind_3,0))
x_tar_4 = tf.convert_to_tensor(np.take(target_X_4,ind_4,0))
y_tar_4 = tf.convert_to_tensor(np.take(target_Y_4,ind_4,0))

x_tar = tf.concat([x_tar_1,x_tar_2,x_tar_3,x_tar_4],0)
y_tar = tf.concat([y_tar_1,y_tar_2,y_tar_3,y_tar_4],0)
###################################################################

# obj = np.empty(n_class,dtype=object)
# for c in range(n_class):
#     obj[i] = np.argwhere(np.round(freq2midi(y_src))==c)[0]

tar = tf.data.Dataset.from_tensor_slices((x_tar, y_tar))
tar = tar.shuffle(y_tar.shape[0]+1, reshuffle_each_iteration=True)

def hausdorffian_distance(xt,y,feat_src,y_src):
    y_src = y_src.numpy()
    y = y.numpy()
    ind_y = np.argwhere(np.round(freq2midi(y_src))==np.round(freq2midi(y)))[0]
    all_ind = np.arange(feat_src.shape[0])
    ind_not_y = np.setxor1d(all_ind, ind_y)

    same_class_max = tf.reduce_max(tf.norm(tf.gather(feat_src,ind_y)-xt,2,axis=1))
    diff_class_min = tf.reduce_min(tf.norm(tf.gather(feat_src,ind_not_y)-xt,2,axis=1))

    return same_class_max-diff_class_min


ind_1 = np.arange(100)
opt = tf.keras.optimizers.Adam(0.0005)
loss_mse = tf.keras.losses.MeanAbsoluteError()

epochs = 15; alpha=0.1; beta=1.0
acc = []
for epoch in range(epochs):
    # tar = tar.shuffle(len_tar+1)
    count=0
    tot_loss = 0
    loss_epoch = None
    np.random.shuffle(ind_1)
    # start = True
    for tar_x,tar_y in tar:
        for src_ind in ind_1[:10]:
            x_src = tf.convert_to_tensor(np.load('data/dsne/src/src_X_t_{}.npy'.format(src_ind)))
            y_src = tf.convert_to_tensor(np.load('data/dsne/src/src_Y_{}.npy'.format(src_ind)))

            with tf.GradientTape() as tape:
                out_src = model(x_src,training=True)
                out_tar = model(x_tar,training=True)
                out_tar_sample = model(tf.reshape(tar_x,[1,tar_x.shape[0]]),training=True)
                
                loss_mse_tot = loss_mse(y_src,out_src[0]) + beta*loss_mse(y_tar,out_tar[0])
                loss_value= (
                            (1-alpha)*loss_mse_tot
                            + alpha*hausdorffian_distance(out_tar_sample[1],tar_y,out_src[1],y_src)
                            )
                
            #     if loss_epoch is None:
            #         loss_epoch = loss_value
            #     else:
            #         loss_epoch = ((loss_epoch*tot_loss)+loss_value)/(tot_loss+1)
            # tot_loss+=1

            grads = tape.gradient(loss_value,model.trainable_weights)
            opt.apply_gradients(zip(grads, model.trainable_weights))
            
            print('epoch',epoch,'target count',count,'source',src_ind)
        
        count+=1
    
    # grads = tape.gradient(loss_epoch,model.trainable_weights)
    # opt.apply_gradients(zip(grads, model.trainable_weights))
    
    rpa,_ = evaluation_dsne(model,target_X,target_Y)
    acc.append(rpa)
    print('############################')
    print('epoch',epoch,'target_RPA',rpa)
    print('############################')

# model.save_weights('saved_models/dsne/model_dsne_weights.h5')
acc.reverse()
acc.append(orig_rpa)
acc.reverse()
acc = np.array(acc)
print(np.max(acc),np.argmax(acc))
plt.plot(np.arange(epochs+1),acc,'o-')
plt.show()

# st = time.time()
# model = Sequential()
# model.add(Dense(1,use_bias=False,input_dim=1,kernel_initializer='zeros'))
# model.add(Dense(1,use_bias=False,input_dim=1,kernel_initializer='zeros'))

# X = np.arange(0,20,0.01)
# train_dataset = tf.data.Dataset.from_tensor_slices((X, X))
# train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

# opt = tf.keras.optimizers.Adam()
# init = model.get_weights()
# loss = tf.keras.losses.MeanSquaredError()
# def custom(a,b):
#     temp = np.reshape(np.copy(a),(a.shape[0],1))
#     a = tf.convert_to_tensor(temp,dtype=tf.float32)
#     return tf.reduce_mean(abs(a-b))
# c=0
# # for epoch in range(100):
# while c<100:
#     # np.random.shuffle(X)
#     # x = X[:64]
#     for x,y in train_dataset:
#         with tf.GradientTape() as tape:
#             out = model(x,training=True)
#             loss_value = custom(y,out)

#         grads = tape.gradient(loss_value,model.trainable_weights)
#         opt.apply_gradients(zip(grads, model.trainable_weights))
#     # print(epoch)
#         c+=1
#         if c==100:
#             break
#         print(c)

# en = time.time()
# print(en-st)





#########################################################
#########################################################

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