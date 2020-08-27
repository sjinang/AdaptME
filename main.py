from library import *

########################################################################
########################################################################

X = np.load('data/data_X.npy')
Y = np.load('data/data_Y.npy')
print(X.shape)

ind = np.where(Y > 1)
Y = Y[ind]
X = X[ind]
print(X.shape)
########################################################################
########################################################################

########################################################################
########################################################################

def main():
    
    kf = KFold(n_splits=5, random_state=1, shuffle=True) # Define the split - into 10 folds 
    kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator

    accuracy = []

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        X_train, X_test = preprocessing_X(X_train, X_test)
        
        model = NN_regressor()
        
        n=1000; b=512
        data_mir.log+='_'+str(n)+'_'+str(b)+'_'
        fitting(model, X_train, Y_train, epochs=n, batch_size=b)
        
        # X_test = transform_X(X_test)
        # X_train = transform_X(X_train)
        # model = load_model('saved_models/model_mir.h5', custom_objects={'RPA': RPA})

        ra_test, Y_pred_test = evaluation(model,X_test,Y_test)
        ra_train,Y_pred_train = evaluation(model,X_train,Y_train)
        
        print(ra_train)
        print(ra_test)
        print()
        accuracy.append([ra_test,ra_train])

        break

    # print(accuracy)


if __name__=='__main__':
    start = time.time()
    main()
    end = time.time()
    print('TOTAL TIME TAKEN',end-start)







