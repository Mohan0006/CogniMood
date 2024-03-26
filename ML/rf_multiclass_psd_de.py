import os
import numpy as np
import pickle as pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

def load_data(file_path):
    with open(file_path, 'rb') as file:
        data = np.load(file)
    return data

def main():
    # Load training and testing data and labels
    X_train = load_data('../data_training_psd_de_32.npy')
    y_train = load_data('../label_training_psd_de_32.npy')
    X_test = load_data('../data_testing_psd_de_32.npy')
    y_test = load_data('../label_testing_psd_de_32.npy')

    # Normalize and standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert labels to numerical values
    Y_train = np.where((y_train[:, 0] >= 5) & (y_train[:, 1] >= 5), 0,
                      np.where((y_train[:, 0] >= 5) & (y_train[:, 1] < 5), 1,
                               np.where((y_train[:, 0] < 5) & (y_train[:, 1] >= 5), 3, 2)))
    
    Y_test = np.where((y_test[:, 0] >= 5) & (y_test[:, 1] >= 5), 0,
                     np.where((y_test[:, 0] >= 5) & (y_test[:, 1] < 5), 1,
                              np.where((y_test[:, 0] < 5) & (y_test[:, 1] >= 5), 3, 2)))
    
    # Print Shapes
    print("Xtrain : ", X_train.shape)
    print("Ytrain : ", Y_train.shape)
    print("Xtest : ", X_test.shape)
    print("Ytest : ", Y_test.shape)
    
    model = RandomForestClassifier(n_estimators=50, random_state=0, max_depth = 30, min_samples_split = 2, min_samples_leaf =1, max_features = 'sqrt')
      
    model.fit(X_train, Y_train)
    
    train_predictions = model.predict(X_train)
    
    train_accuracy = accuracy_score(Y_train,train_predictions)
    
    print("Training accuracy:", train_accuracy)
    

    # Make predictions on the test set
    test_predictions = model.predict(X_test)

    # Evaluate the model on the test set
    test_accuracy = accuracy_score(Y_test, test_predictions)
    print("Test accuracy:", test_accuracy)
    

if __name__ == "__main__":
    main()
