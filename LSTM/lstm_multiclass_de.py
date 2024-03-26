import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.optimizers import Adam

def create_and_compile_model(input_shape, num_classes):
    # Create a Sequential model for multi-class classification
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.6))
    model.add(LSTM(units=256, return_sequences=True))
    model.add(Dropout(0.6))
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.6))
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(units=32))
    model.add(Dropout(0.4))
    model.add(Dense(units=16))
    model.add(Activation('relu'))
    model.add(Dense(units=num_classes))
    model.add(Activation('softmax'))
    
    # Compile the model with the Adam optimizer and categorical cross-entropy loss
    optimizer = Adam()  # You can adjust the learning rate as needed
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    
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

    y_train = []
    
    for i in range(Y_train.shape[0]):
      if Y_train[i][0] >= 5 and Y_train[i][1] >= 5:
        y_train.append(0)
      elif Y_train[i][0] >= 5 and Y_train[i][1] < 5:
        y_train.append(3)
      elif Y_train[i][0] < 5 and Y_train[i][1] >= 5:
        y_train.append(1)
      else:
        y_train.append(2)
        
    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train)
    
    # Load testing data and labels
    with open('../data_testing_de.npy', 'rb') as fileTest:
        X_test = np.load(fileTest)
        
    with open('../label_testing_de.npy', 'rb') as fileTestL:
        Y_test = np.load(fileTestL)

    # Normalize the testing data
    X_test = normalize(X_test)


    
    y_test = []
    
    for i in range(Y_test.shape[0]):
      if Y_test[i][0] >= 5 and Y_test[i][1] >= 5:
        y_test.append(0)
      elif Y_test[i][0] >= 5 and Y_test[i][1] < 5:
        y_test.append(3)
      elif Y_test[i][0] < 5 and Y_test[i][1] >= 5:
        y_test.append(1)
      else:
        y_test.append(2)
        
    # Convert labels to one-hot encoding
    y_test = to_categorical(y_test)    
    

    # Standardize the data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(X_train)
    x_test = scaler.fit_transform(X_test)

    # Reshape data for Conv1D
    input_shape = (x_train.shape[1], 1)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    
    # Print Shapes
    print("Xtrain : ", x_train.shape)
    print("Ytrain : ", y_train.shape)
    print("Xtest : ", x_test.shape)
    print("Ytest : ", y_test.shape)
    
    # Model parameters
    batch_size = 1024
    num_classes = len(y_train[0])
    epochs = 200

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
  
    # Create and compile the model for Valence
    model = create_and_compile_model(input_shape, num_classes)
    
    # Train the model for Arousal
    history = model.fit(x_train, y_train,
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        verbose=1,
                                        validation_data=(x_test, y_test),
                                        callbacks=[early_stopping])
                                       
    # model = load_model('cnn_multi_class_classifier.h5')


    
    # Print model summary
    model.summary()
    
    # Access the training and validation loss and accuracy from the history object
    training_loss = history.history['loss']
    training_accuracy = history.history['accuracy']
    validation_loss = history.history['val_loss']
    validation_accuracy = history.history['val_accuracy']

    # Print the final metrics
    final_epoch = len(training_loss)
    print(f'Final Training Loss: {training_loss[-1]}, Training Accuracy: {training_accuracy[-1]}')
    print(f'Final Validation Loss: {validation_loss[-1]}, Validation Accuracy: {validation_accuracy[-1]}')
    
    plot_model(model, to_file='model.png', show_shapes=True)
    
    # Plot the loss curve
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Loss Curve - Multi_Class_Classification_Valence_Arousal_Space (PSD)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Save the model
    model.save('lstm_multi_class_classifier.h5')
    

if __name__ == "__main__":
    main()
