from library import *
from sklearn.metrics import accuracy_score
np.random.seed(111)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#################################################
#################################################

def load_data():

    X = np.load('data/data_X.npy')
    Y = np.load('data/data_Y.npy')
    print(Y.shape)

    ind = np.where(Y > 1)
    Y = Y[ind]
    X = X[ind]
    # print(np.min(Y))
    Y = freq2midi(Y)
    print(np.min(Y.round()))
    Y1 = Y.round()-np.min(Y.round())
    print(np.min(Y1),np.max(Y1))
    
    Y1 = (tf.one_hot(Y1,int(np.max(Y1)-np.min(Y1)+1))).numpy()
    print(Y1[np.argmin(Y)])
    print(Y1[np.argmax(Y)])
    print(X.shape,Y1.shape)
    
    return X, Y, Y1


def preprocessing_X(X_train,X_test):
    
    scaler = Normalizer().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, 'saved_models/normalizer_classreg.pkl') 

    scaler1 = StandardScaler()
    scaler1.fit(X_train)
    joblib.dump(scaler1, 'saved_models/standardscaler_classreg.pkl') 

    X_train = scaler1.transform(X_train)
    X_test = scaler1.transform(X_test)

    return X_train, X_test


def NN():

    model = Sequential()
    model.add(Input((1025,)))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='tanh'
                    # ,kernel_regularizer=l1(0.001)
                    ))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='tanh'
                    # ,kernel_regularizer=l1(0.00001)
                    ))
    model.add(Dropout(0.2))
    # model.add(Dense(512, activation='tanh'))
    # model.add(Dropout(0.2))
    model.add(Dense(41, activation='softmax'))

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer='adam', metrics=['categorical_accuracy'])
    
    return model


#################################################
#################################################

def main():

    X, Y1, Y = load_data()

    X = X[:,:512]

    kf = KFold(n_splits=3, random_state=1, shuffle=True) # Define the split - into 10 folds 
    kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator

    accuracy = []

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        Y1_train, Y1_test = Y1[train_index], Y1[test_index]

        X_train, X_test = preprocessing_X(X_train, X_test)

        model = NN_classifier()
        model.fit(X_train,Y_train,1024,50)
        model.save('saved_models/model_class.h5')
        result_test = model.evaluate(X_test,Y_test,X_test.shape[0])
        print(result_test)
        Y_pred = np.argmax(model.predict(X_test), axis=-1)
        Y_pred_midi = (Y_pred/1).round()
        print('RPA :',accuracy_score(Y1_test.round()-np.min(Y1_test.round()),Y_pred_midi))

 
        break


#################################################
#################################################

if __name__=='__main__':
    start = time.time()
    main()
    end = time.time()
    print('TOTAL TIME TAKEN',end-start)




