# Goals:
# #1: load in the HOPE pitch angle data
# #2: calculate 90 deg / (0 deg + 180 deg) electron temp ani for all energy channels
# #3: interpolate the data to the

def get_hope(vlf_time):
  from os.path import dirname, join as pjoin
  import scipy.io as sio
  from scipy.io import readsav
  import os
  import numpy as np
  import pandas as pd
  import time
  import datetime
  from scipy.interpolate import interp1d
### check content of the folder
  foldername = r'D:\Research\Data\Van_Allen_Probes\HOPE_L3_mat\\'
  folder_content=[f for f in os.listdir(foldername) if not f.startswith('.')]
  folder_content.sort()


  hope=[]
  u=0
  for i in range(0,len(folder_content)):
    mat_contents = sio.loadmat(foldername + folder_content[i]) # load in mat
    datenums = np.squeeze(np.array(mat_contents['HOPE'][0][0]),axis=1) # convert mat time to unix time
    timestamps = pd.to_datetime(datenums-719529, unit='D')
    unixtime=[]
    for p in range(0,len(timestamps)):
      unixtime.append(time.mktime(timestamps[p].timetuple()))
    time_grid0=np.array(unixtime)
  #### grid with temp anisotropy as a function of time (row) and energy (column, increasing)
    #e_ani0=(mat_contents['HOPE'][0][1][5,:,:]/(mat_contents['HOPE'][0][1][1,:,:]+mat_contents['HOPE'][0][1][9,:,:])).T
    e_perp=0.33*(mat_contents['HOPE'][0][1][4,:,:]+mat_contents['HOPE'][0][1][5,:,:]+mat_contents['HOPE'][0][1][6,:,:]).T
    e_par=0.25*(mat_contents['HOPE'][0][1][1,:,:]+mat_contents['HOPE'][0][1][2,:,:]+mat_contents['HOPE'][0][1][9,:,:]+mat_contents['HOPE'][0][1][8,:,:]).T
    e_ani0=e_perp/e_par
    e_ani0[~np.isfinite(np.abs(e_ani0))] = np.nan
    e_range0=mat_contents['HOPE'][0][3]
  

    if u==0:
      hope.append(time_grid0) #time
      hope.append(e_ani0) #electron temp anisotropy
      hope.append(e_range0) #energy level
      u=u+1
    else:
      hope[0]=np.concatenate([ hope[0],time_grid0 ],axis=0) #time
      hope[1]=np.concatenate([ hope[1], e_ani0 ],axis=0) #e anisotropy
      hope[2]=np.concatenate([ hope[2], e_range0 ],axis=0) #energy level
    print(str(i))   

# delete all of those times when the energy range was incorrect 
  bad_hope=np.argwhere(np.diff(hope[2][:,0])>0)  
  hope[0]=np.delete(hope[0],bad_hope)
  hope[1]=np.delete(hope[1],bad_hope,axis=0)
  hope[2]=np.delete(hope[2],bad_hope,axis=0)
# interpolate hope to vlf_time
 
  f_interp = interp1d( hope[0], hope[1].T,fill_value='nan',bounds_error=False)
  hope_interp = f_interp(vlf_time).T
  
  hope_interp[hope_interp==0]='nan'
  
  hope_energy=hope[2][0,:].T
  return hope_interp, hope_energy
     