import os
import numpy as np
import pickle as pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize

def main():
    
    # Load training data and labels
    with open('../data_training_psd_32.npy', 'rb') as fileTrain:
        X_train = np.load(fileTrain)
        
    with open('../label_training_psd_32.npy', 'rb') as fileTrainL:
        Y_train = np.load(fileTrainL)

    # Normalize the training data
    X_train = normalize(X_train)

    y_train_valence = []
    y_train_arousal = []
    
    for i in range(Y_train.shape[0]):
      if Y_train[i][0] >= 5 and Y_train[i][1] >= 5:
        y_train_valence.append(1)
        y_train_arousal.append(1)
      elif Y_train[i][0] >= 5 and Y_train[i][1] < 5:
        y_train_valence.append(1)
        y_train_arousal.append(0)
      elif Y_train[i][0] < 5 and Y_train[i][1] >= 5:
        y_train_valence.append(0)
        y_train_arousal.append(1)
      else:
        y_train_valence.append(0)
        y_train_arousal.append(0)
        

    # Load testing data and labels
    with open('../data_testing_psd_32.npy', 'rb') as fileTest:
        X_test = np.load(fileTest)
        
    with open('../label_testing_psd_32.npy', 'rb') as fileTestL:
        Y_test = np.load(fileTestL)

    # Normalize the testing data
    X_test = normalize(X_test)


    
    y_test_valence = []
    y_test_arousal = []
    
    for i in range(Y_test.shape[0]):
      if Y_test[i][0] >= 5 and Y_test[i][1] >= 5:
        y_test_valence.append(1)
        y_test_arousal.append(1)
      elif Y_test[i][0] >= 5 and Y_test[i][1] < 5:
        y_test_valence.append(1)
        y_test_arousal.append(0)
      elif Y_test[i][0] < 5 and Y_test[i][1] >= 5:
        y_test_valence.append(0)
        y_test_arousal.append(1)
      else:
        y_test_valence.append(0)
        y_test_arousal.append(0)
    
    y_train_valence = np.array(y_train_valence)
    y_test_valence = np.array(y_test_valence)    
    

    # Standardize the data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(X_train)
    x_test = scaler.fit_transform(X_test)

    # Reshape data for Conv1D
    X_train = x_train
    X_test = x_test
    Y_train = y_train_valence
    Y_test = y_test_valence
    
    # Convert labels to one-hot encoding
    Y_train = to_categorical(Y_train) 
    Y_test = to_categorical(Y_test)  
    
    
    # Print Shapes
    print("Xtrain : ", X_train.shape)
    print("Ytrain : ", Y_train.shape)
    print("Xtest : ", X_test.shape)
    print("Ytest : ", Y_test.shape)
    
    model = RandomForestClassifier(n_estimators=50, random_state=0, max_depth = 30, min_samples_split = 2, min_samples_leaf =1, max_features = 'sqrt')

    model.fit(X_train, Y_train)
    
    # Make predictions on the test set
    predictions = model.predict(X_test)
    
    train_pred = model.predict(X_train)
    
    train_accuracy = accuracy_score(Y_train, train_pred)
    print("Train accuracy:", train_accuracy)
    
    test_accuracy = accuracy_score(Y_test, predictions)
    print("Test accuracy:", test_accuracy)
    
if __name__ == "__main__":
    main()