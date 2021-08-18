
# delete bad data, fix time stamps

import numpy as np
import pandas as pd

def del_bad_data(vlf0):
  A=[]
  err=np.where( (vlf0['eclipse_flag']>0) | (vlf0['thruster_flag']>0) | (vlf0['charging_flag']>0) )[0]
  np.delete(vlf0['b_spectra'],err, axis=1)
  A.append(vlf0['b_spectra'])
  
  np.delete(vlf0['e_spectra_spin'],err, axis=1)
  A.append(vlf0['e_spectra_spin'])
  
  np.delete(vlf0['e_spectra_axial'],err, axis=1)
  A.append(vlf0['e_spectra_axial'])
  
  np.delete(vlf0['compressability'],err, axis=1)
  A.append(vlf0['compressability'])

  np.delete(vlf0['b_time'],err, axis=0)
  
  np.delete(vlf0['b_mag'],err, axis=0)
  A.append(vlf0['b_mag'])
  
  NN = 719529 # NN = datenum('01-jan-1970 00:00:00','dd-mmm-yyyy HH:MM:SS')    
  #b_time = (vlf0['b_time']/86400) + NN # old
  b_time = vlf0['b_time'] # new
  vlf0['b_time'] = b_time # new
  #vlf0['b_time'] = pd.to_datetime(b_time-719529, unit='D') # old
  A.append(vlf0['b_time'])
  
  A.append(vlf0['unixtime'])
  A.append(vlf0['mlt'])
  A.append(vlf0['mlat'])
  A.append(vlf0['lshell'])
  
  return A # insteaf of vlf0