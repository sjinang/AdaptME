from utils import *

########################################################################
########################################################################




def NN_regressor_0b(input_dim=1025,output_dim=1):
    # model = Sequential()

    # # model.add(Dropout(0, input_shape=(1025,)))
    # model.add(Input((1025,)))
    # model.add(Dropout(0.2))
    # n=256; data_mir.log=str(n); model.add(Dense(n
    #                                             , activation='tanh'
    #                                             # , kernel_regularizer=l1(0.0005)
    #                                             )) # ^softsign,tanh
    # model.add(Dropout(0.2))

    # n=256; model.add(Dense(n, activation='tanh'))
    # # model.add(Dropout(0.2))
    
    # n=256;  model.add(Dense(n, activation='tanh'))
    # # model.add(Dropout(0.2))
    
    # # n=256; model.add(Dense(n, activation='tanh'))
    
    # # n=500; model.add(Dense(n, activation='tanh',kernel_regularizer=keras.regularizers.l1()))
    
    # model.add(Dense(1, activation='relu'))
    
    # model.compile(loss='mean_absolute_error'
    #             , optimizer='adam'
    #             , metrics=['mae',RPA]
    #             ) 

    x_in = Input((input_dim,))
    x = Dropout(0.2)(x_in)
    x = Dense(256, 'tanh')(x)
    # x = Dropout(0.2)(x)
    x = Dense(256, 'tanh')(x)

    feat = Dense(256, 'tanh')(x)
    pred = Dense(1,'relu')(feat)

    model = Model(x_in,pred)
    model.compile('adam','mae')  

    return model

def NN_regressor_0(input_dim=1025,output_dim=1):

    x_in = Input((input_dim,))

    x_at = tf.keras.layers.MultiHeadAttention(num_heads=1, key_dim=128, attention_axes=1, dropout=0.0)(x_in,x_in)

    x = Dropout(0.2)(x_in)
    
    x = Dense(256, 'tanh')(x_at)
    
    x = Dense(256, 'tanh')(x)

    feat = Dense(256, 'tanh')(x)
    
    pred = Dense(1,'relu')(feat)

    model = Model(x_in,pred)
    model.compile('adam','mae')  

    return model


def NN_regressor_dsne(input_dim=1025,output_dim=1):
    x_in = Input((input_dim,))
    x = Dropout(0.2)(x_in)
    x = Dense(256, 'tanh')(x)
    # x = Dropout(0.2)(x)
    x = Dense(256, 'tanh')(x)

    feat = Dense(256, 'tanh')(x)
    pred = Dense(1,'relu')(feat)

    model = Model(x_in,[pred,feat])
    model.compile('adam','mae',['mae',RPA])  

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


def NN_classifier_dsne(input_dim=1025):

    x_in = Input((input_dim,))
    x = Dropout(0.2)(x_in)
    x = Dense(256, activation='tanh'
                    # ,kernel_regularizer=l1(0.001)
                    )(x)
    x = Dropout(0.2)(x)
    feat = Dense(256, activation='tanh'
                    # ,kernel_regularizer=l1(0.001)
                    )(x)
    x = Dropout(0.2)(feat)
    # model.add(Dense(512, activation='tanh'))
    # model.add(Dropout(0.2))
    pred = Dense(90-33+1, activation='softmax')(x)

    model = Model(x_in,[pred,feat])
    model.compile(loss=[tf.keras.losses.SparseCategoricalCrossentropy(),None], optimizer='adam', metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    
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
