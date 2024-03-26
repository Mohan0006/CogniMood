import numpy as np

data_training = []
label_training = []
data_testing = []
label_testing = []

subjectList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]  # Replace with your subject list

# subjectList = [1, 2, 3, 4, 5]

for subject in subjectList:

    with open(f'./DE_Data/s{subject}_de.npy', 'rb') as de_file:
        de_data = np.load(de_file, allow_pickle=True)

        for i in range(de_data.shape[0]):
        
           if i % 10 == 0:
             data_testing.append(de_data[i][0])
             label_testing.append(de_data[i][1])  # Corrected the label indexing
           else:
             data_training.append(de_data[i][0])
             label_training.append(de_data[i][1])  # Corrected the label indexing

# Convert lists to numpy arrays
data_training = np.array(data_training)
label_training = np.array(label_training)
data_testing = np.array(data_testing)
label_testing = np.array(label_testing)

# Save the merged data to separate files
np.save('data_training_de', data_training)
np.save('label_training_de', label_training)
print("Training dataset:", data_training.shape, label_training.shape)

np.save('data_testing_de', data_testing)
np.save('label_testing_de', label_testing)
print("Testing dataset:", data_testing.shape, label_testing.shape)
