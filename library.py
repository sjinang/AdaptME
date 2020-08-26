from import_libs import *

########################################################################
########################################################################

def RPA(y_true,y_pred):
    y_true = ((12*tf.math.log(y_true/440)) / tf.math.log( tf.constant(2,dtype=tf.float32) )) + 69
    y_pred = ((12*tf.math.log(y_pred/440)) / tf.math.log( tf.constant(2,dtype=tf.float32) )) + 69
    
    a = tf.abs(y_true-y_pred) < 0.5
    a = tf.cast(a,dtype=tf.float32)
    return 100*tf.reduce_mean(a)


def NN_regressor():
    model = Sequential()

    # model.add(Dropout(0, input_shape=(1025,)))
    model.add(Input((1025,)))
    model.add(Dropout(0.2))
    n=256; data_mir.log=str(n); model.add(Dense(n
                                                , activation='relu'
            
                                                # , kernel_regularizer=l1(0.0005)
                                                )) # ^softsign,tanh
    
    n=256; data_mir.log+='x'+str(n); model.add(Dense(n, activation='relu'))
    model.add(Dropout(0.2))
    n=256; data_mir.log+='x'+str(n); model.add(Dense(n, activation='relu'))
    model.add(Dropout(0.2))
    # n=256; data_mir.log+='x'+str(n); model.add(Dense(n, activation='tanh'))
    # n=500; data_mir.log+='x'+str(n); model.add(Dense(n, activation='tanh',kernel_regularizer=keras.regularizers.l1()))
    
    model.add(Dense(1, activation='relu'))
    
    model.compile(loss='mean_absolute_error'
                , optimizer='adam'
                , metrics=['mae',RPA]
                ) 

    return model
    
########################################################################
########################################################################

def NN_classifier():
    model = Sequential()
    model.add(Dense(100,input_dim=1025,activation='tanh',kernel_regularizer=l1(0.0001)))
    model.add(Dense(100,activation='tanh'))
    model.add(Dense(100,activation='tanh'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='Adam',metrics=['accuracy']) 

    return model

########################################################################
########################################################################
def NMF(X_train, n_comp):

    model = sklearn.decomposition.NMF(n_components=n_comp,
                                        init='random',
                                        random_state=1,
                                        max_iter=500,
                                        verbose=True,
                                        tol=5e-5)
    W = model.fit_transform(X_train)
    H = model.components_   

    return W, H

########################################################################
########################################################################
