import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from library import *
# a=tf.constant([1.0,2.0,3.0])
# b=tf.constant([[0.0],[-1.0],[6.0]])
# loss= tf.keras.losses.MeanSquaredError()
# print(loss(a,b).numpy())

# A = np.arange(10)
# B = np.array([1,3,4,7,9])

# print(A[B[np.searchsorted(B,A)] !=  A])

# a = tf.constant([[1,2],[3,6],[1,0]])
# b = tf.constant([[15,8]])
# temp = np.tile(np.square(np.linalg.norm(a,axis=1)),(b.shape[0],1)) + np.tile(np.square(np.linalg.norm(b,axis=1)),(a.shape[0],1)).T - (2*(np.matmul(b,a.T)))
# print(tf.reduce_max(tf.norm(a-b,1,axis=1)))
# print(tf.gather(a,np.array([1])))

b = np.array([1,2,3,4,5,6,7])
print((b>3)*(b<6))