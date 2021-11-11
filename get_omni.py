
# -Goal: load in OMNI data, interpolate it to the times in vlf
# -return interpolated data set


def get_omni(vlf_time):
 
  from os.path import dirname, join as pjoin
  import scipy.io as sio
  from scipy.io import readsav
  import os
  import numpy as np
  import pandas as pd
  import time
  import datetime
  from scipy.interpolate import interp1d

  mat_contents = sio.loadmat(r'D:\Research\Data\OMNI\OMNI.mat') # load in mat
  #datenums = np.squeeze(np.array(mat_contents['OMNI'][:,0]),axis=1) # convert mat time to unix time
  datenums = np.array(mat_contents['OMNI'][:,0]) #
  timestamps = pd.to_datetime(datenums-719529, unit='D')
  unixtime=[]
  for p in range(0,len(timestamps)):
    unixtime.append(time.mktime(timestamps[p].timetuple()))
  time_grid0=np.array(unixtime)
  #### grid with temp anisotropy as a function of time (row) and energy (column, increasing)
  f_interp = interp1d( time_grid0, mat_contents['OMNI'].T,fill_value='nan',bounds_error=False,kind='nearest')
  omni_interp = f_interp(vlf_time).T

  return omni_interp    
    
    