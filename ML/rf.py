import os
import numpy as np
import pickle as pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
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

    print("..")

    # Define the parameter grid for Grid Search
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    # Create the RandomForestClassifier
    model = RandomForestClassifier(random_state=0)

    # Perform Grid Search
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, Y_train)

    # Get the best parameters
    best_params = grid_search.best_params_

    print("Best Hyperparameters:", best_params)

    # Train the model with the best hyperparameters
    model = RandomForestClassifier(**best_params, random_state=0)
    model.fit(X_train, Y_train)

    # Make predictions on the training set
    train_predictions = model.predict(X_train)

    # Evaluate the model on the training set
    train_accuracy = accuracy_score(Y_train, train_predictions)
    print("Training accuracy:", train_accuracy)

    # Make predictions on the test set
    test_predictions = model.predict(X_test)

    # Evaluate the model on the test set
    test_accuracy = accuracy_score(Y_test, test_predictions)
    print("Test accuracy:", test_accuracy)


if __name__ == "__main__":
    main()
