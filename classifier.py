from library import *
from sklearn.metrics import accuracy_score
np.random.seed(111)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#################################################
#################################################
minm = 33
maxm = 90
def load_data():

    X = np.load('data/data_X.npy')[:,:512]
    Y = np.load('data/data_Y.npy')
    print(Y.shape)

    ind = np.where(Y > 1)
    Y = Y[ind]
    X = X[ind]
    # print(np.min(Y))
    Y = freq2midi(Y)
    print(np.min(Y.round()))
    # Y1 = Y.round()-np.min(Y.round())
    # print(np.min(Y1),np.max(Y1))

    Y1 = Y.round() - minm
    
    # Y1 = (tf.one_hot(Y1,maxm-minm+1)).numpy()
    # print(Y1[np.argmin(Y)])
    print(np.max(Y1))
    print(X.shape,Y1.shape)
    
    return X, Y, Y1


def preprocessing_X(X_train,X_test):
    
    scaler = Normalizer().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, 'saved_models/normalizer_class.pkl') 

    scaler1 = StandardScaler()
    scaler1.fit(X_train)
    joblib.dump(scaler1, 'saved_models/standardscaler_class.pkl') 

    X_train = scaler1.transform(X_train)
    X_test = scaler1.transform(X_test)

    return X_train, X_test


def NN_classifier(input_dim=1025):

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
    pred = Dense(maxm-minm+1, activation='softmax')(x)

    model = Model(x_in,pred)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    
    return model


#################################################
#################################################

def main():

    X, Y, Y1 = load_data()

    kf = KFold(n_splits=5, random_state=1, shuffle=True) # Define the split - into 10 folds 
    kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator

    accuracy = []

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        Y1_train, Y1_test = Y1[train_index], Y1[test_index]

        X_train, X_test = preprocessing_X(X_train, X_test)
        # print(np.max(Y1_train))
        model = NN_classifier(512)
        model.fit(X_train,Y1_train,512,25,validation_split=0.1)
        model.save('saved_models/model_class_dsne.h5')

        # model = load_model("saved_models/model_class_dsne.h5")

        # result_test = model.evaluate(X_test,Y_test,X_test.shape[0])
        # print(result_test)
        
        Y_pred_c = np.argmax(model.predict(X_test), axis=-1)
        # Y1_test_c = np.argmax(Y1_test, axis=-1)
        print('RPA :',accuracy_score(Y1_test,Y_pred_c))

 
        break


#################################################
#################################################

if __name__=='__main__':
    start = time.time()
    main()
    end = time.time()
    print('TOTAL TIME TAKEN',end-start)




