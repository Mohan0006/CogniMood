import os
import numpy as np
import pickle as pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier

def main():
    # Load training data and labels
    with open('../data_training_de.npy', 'rb') as fileTrain:
        X_train = np.load(fileTrain)
        
    with open('../label_training_de.npy', 'rb') as fileTrainL:
        y_train = np.load(fileTrainL)

    # Load testing data and labels
    with open('../data_testing_de.npy', 'rb') as fileTest:
        X_test = np.load(fileTest)
        
    with open('../label_testing_de.npy', 'rb') as fileTestL:
        y_test = np.load(fileTestL)

    # Normalize and standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    Y_train = []
    Y_test = []
    
    
    for i in range(y_train.shape[0]):
      if y_train[i][0] >= 5 and y_train[i][1] >= 5:
        Y_train.append(0)
      elif y_train[i][0] >= 5 and y_train[i][1] < 5:
        Y_train.append(1)
      elif y_train[i][0] < 5 and y_train[i][1] >= 5:
        Y_train.append(3)
      else:
        Y_train.append(2)    
    
    for i in range(y_test.shape[0]):
      if y_test[i][0] >= 5 and y_test[i][1] >= 5:
        Y_test.append(0)
      elif y_test[i][0] >= 5 and y_test[i][1] < 5:
        Y_test.append(1)
      elif y_test[i][0] < 5 and y_test[i][1] >= 5:
        Y_test.append(3)
      else:
        Y_test.append(2)
    
    Y_test = np.array(Y_test)
    Y_train = np.array(Y_train)
    
    
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
