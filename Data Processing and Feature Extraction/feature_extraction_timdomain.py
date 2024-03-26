import _pickle as cPickle
import csv
import pandas as pd
import numpy as np

def extract_nsi_feature(data):
    # compute the mean of the signal
    x = np.mean(data)
    
    # compute the standard deviation of the signal
    sigma = np.std(data)
    
    # compute the skewness of the signal
    skewness = np.mean((data - x)*3) / (sigma*3)
    
    # compute the kurtosis of the signal
    kurtosis = np.mean((data - x)*4) / (sigma*4) - 3.0
    
    # compute the NSI feature
    nsi = (skewness*2 + kurtosis*2) / 2.0
    
    return nsi


for i in range(32):
   if i+1 < 10:
    x = cPickle.load(open("./data/s0"+str(i+1)+".dat", 'rb'), encoding="bytes")
   else:
    x = cPickle.load(open("./data/s"+str(i+1)+".dat", 'rb'), encoding="bytes")

   field_names = []
   for key in x.keys():
      field_names.append(key)

   labels = x[field_names[0]]
   data = x[field_names[1]]

   lst = [0, 16, 2, 24]
   dat = []
   for i in range(40):
        tmp = []
        for j in lst:
            tmp.append(data[i][j])
        dat.append(tmp)
   dat = np.array(dat)
   feature = []
   for ch_idx in range(dat.shape[1]):
        channel_data = dat[:, ch_idx, :]  # select data for current channel
    
         # calculate variance of the signal
        var = np.var(channel_data, axis=1)
    
        # calculate first and second order differences
        diff1 = np.diff(channel_data, axis=1)
        diff2 = np.diff(diff1, axis=1)
    
        # calculate mobility and complexity
        
        mobility = np.sqrt(np.divide(np.var(diff1, axis=1), var))
        complexity = np.sqrt(np.divide(np.var(diff2, axis=1), np.var(diff1, axis=1))) / mobility
    
        tmp = []
        tmp.append(var)
        tmp.append(mobility)
        tmp.append(complexity)
        feature.append(tmp)
    
   feature = np.array(feature)
   feature = feature.transpose((2, 1, 0))
   tmp = feature
   feature = []
   for i in range(40):
       lst = []
       for j in range(3):
          val = np.mean(tmp[i][j])
          lst.append(val)
       # calculate NSI feature
          nsi_feature = extract_nsi_feature(tmp[i][0])

       lst.append(np.mean(nsi_feature))
       feature.append(lst) 
   feature = np.array(feature)
   
   
   # Combine the arrays horizontally
   combined_array = np.concatenate((feature, labels[:,0:2]), axis=1)
   df = pd.DataFrame(combined_array)
   df.columns = ["Activity","Mobility","Complexity","NSI","Valence","Arousal"]
   
   df.to_csv('features.csv', mode='a', header=False, index=False)