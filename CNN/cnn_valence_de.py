import os
import numpy as np
import pickle as pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import (
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import EarlyStopping

def create_and_compile_model(input_shape, num_classes):
    model = Sequential()
    
    # Convolutional layers with increased filter size and stride
    model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Conv1D(64, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    
    # Flattening and Dense layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))  # Increase the number of units
    model.add(Dropout(0.5))  # Adjust dropout rate
    
    model.add(Dense(128, activation='relu'))  # Additional dense layer
    model.add(Dropout(0.4))
    
    model.add(Dense(64, activation='relu'))  # Additional dense layer
    model.add(Dropout(0.3))
    
    model.add(Dense(num_classes, activation='softmax'))

    
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001),  metrics=['accuracy'])
    
    return model

def main():
    # Set the GPU visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Load training data and labels
    with open('../data_training_de.npy', 'rb') as fileTrain:
        X_train = np.load(fileTrain)
        
    with open('../label_training_de.npy', 'rb') as fileTrainL:
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
    with open('../data_testing_de.npy', 'rb') as fileTest:
        X_test = np.load(fileTest)
        
    with open('../label_testing_de.npy', 'rb') as fileTestL:
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
    input_shape = (x_train.shape[1], 1)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    y_train_valence = y_train_valence.reshape(-1,1)
    y_test_valence = y_test_valence.reshape(-1,1)
    
    # Convert labels to one-hot encoding
    y_train_valence = to_categorical(y_train_valence) 
    y_test_valence = to_categorical(y_test_valence)  
    
    # Print Shapes
    print("Xtrain : ", x_train.shape)
    print("Ytrain : ", y_train_valence.shape)
    print("Xtest : ", x_test.shape)
    print("Ytest : ", y_test_valence.shape)
    
    # Model parameters
    batch_size = 1024
    num_classes = len(y_train_valence[0])
    epochs = 200

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    # Create and compile the model for Valence
    model_valence = create_and_compile_model(input_shape, num_classes)
    
    # Train the model for Arousal
    history = model_valence.fit(x_train, y_train_valence,
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        verbose=1,
                                        validation_data=(x_test, y_test_valence),
                                        callbacks=[early_stopping])
    # Access the training and validation loss and accuracy from the history object
    training_loss = history.history['loss']
    training_accuracy = history.history['accuracy']
    validation_loss = history.history['val_loss']
    validation_accuracy = history.history['val_accuracy']

    # Print the final metrics
    final_epoch = len(training_loss)
    print(f'Final Training Loss: {training_loss[-1]}, Training Accuracy: {training_accuracy[-1]}')
    print(f'Final Validation Loss: {validation_loss[-1]}, Validation Accuracy: {validation_accuracy[-1]}')
    
    # Print model summary
    model_valence.summary()
    
    # Plot the loss curve
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('CNN Loss Curve - Binary_Classification_Valence (DE)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Save the model
    model_valence.save('cnn_valence_classifier.h5')


if __name__ == "__main__":
    main()
