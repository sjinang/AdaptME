import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from library import *

X = np.load('data/data_X_1.npy')[:,:512]
Y = np.load('data/data_Y_1.npy')
print(X.shape)
# print(Y[:10])
Y1 = freq2midi(Y).round()
# y = np.loadtxt('mirex05/train10REF.txt')[:,1]
print(Y1.shape[0])

# X = np.copy(X0)
# Y = np.copy(Y0)
frac = 0.01
for epoch in range(int(1/frac)):    
    places=[]
    for l in range(100):
        ind = np.where(Y1==l)[0]
        temp = ind.shape[0]
        if ind.shape[0]==0: 
            continue
        ext = int(np.ceil(frac*ind.shape[0]))
        np.random.shuffle(ind)
        places.extend(ind[:ext])
        # X = np.delete(X,ind,0)
        # Y = np.delete(Y,ind,0)
        # Y1 = np.delete(Y1,ind,0)
        # print(l,temp,ext,Y.shape[0])
    places = np.array(places)
    np.random.shuffle(places)
    np.save('data/dsne/src/src_X_t_{}.npy'.format(epoch),transform_X(np.take(X,places,0)))
    np.save('data/dsne/src/src_Y_{}.npy'.format(epoch),np.take(Y,places,0))
    print(epoch,places.shape[0])

# plt.hist(Y1,np.arange(35,90),histtype='step')
# plt.hist(y1,np.arange(35,90),histtype='step')
# plt.show()