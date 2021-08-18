# 1) use the pre-trained SOM model on the cleaned spin data

def spin_som(vlf):

  from os.path import dirname, join as pjoin
  import scipy.io as sio
  from scipy.io import readsav
  import os
  import numpy as np
  import pickle
## normalize data to ~2Hz frequency and drop some frequency bins above 2kHz
  vlf_norm=[]
  vlf_norm.append(np.log10(vlf[0][0:50,:])-np.log10(vlf[0][0,:]))
  vlf_norm.append(np.log10(vlf[1][0:50,:])-np.log10(vlf[1][0,:]))
  vlf_norm.append(np.log10(vlf[2][0:50,:])-np.log10(vlf[2][0,:]))    
  
  filename=(r'D:\Research\Data\Van_Allen_Probes\Models\model_spin.sav')
  vlf_som = pickle.load(open(filename, 'rb'))
  predictions = vlf_som.predict(vlf_norm[1].T) #[1] spin data
  
  return predictions, vlf_norm