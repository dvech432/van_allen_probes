# Goals:
#1 load in MagEIS data
#2 calculate the e anisotropy for each energy bins
#3 interpolate to vlf time

def get_mageis(vlf_time):
    
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
  foldername = r'D:\Research\Data\Van_Allen_Probes\MagEIS_mat\\'
  folder_content=[f for f in os.listdir(foldername) if not f.startswith('.')]
  folder_content.sort()    

  mageis=[]
  u=0
  for i in range(0,len(folder_content)):
    mat_contents = sio.loadmat(foldername + folder_content[i]) # load in mat
    datenums = np.squeeze(np.array(mat_contents['MagEIS'][0][0]),axis=1) # convert mat time to unix time
    timestamps = pd.to_datetime(datenums-719529, unit='D')
    unixtime=[]
    for p in range(0,len(timestamps)):
      unixtime.append(time.mktime(timestamps[p].timetuple()))
    time_grid0=np.array(unixtime)
  #### grid with temp anisotropy as a function of time (row) and energy (column, increasing)
    #e_ani0=(mat_contents['HOPE'][0][1][5,:,:]/(mat_contents['HOPE'][0][1][1,:,:]+mat_contents['HOPE'][0][1][9,:,:])).T
    e_perp=mat_contents['MagEIS'][0][1][5,:,:]
    e_par=(mat_contents['MagEIS'][0][1][1,:,:]+mat_contents['MagEIS'][0][1][9,:,:])/2
    e_ani0=(e_perp/e_par).T
    e_ani0[~np.isfinite(np.abs(e_ani0))] = np.nan
    e_ani0[(e_ani0)<0] = np.nan
    e_ani0[(e_ani0)==0] = np.nan
    e_ani0[(e_ani0)==1] = np.nan
    e_range0=mat_contents['MagEIS'][0][2]
  

    if u==0:
      mageis.append(time_grid0) #time
      mageis.append(e_ani0) #electron temp anisotropy
      mageis.append(e_range0) #energy level
      u=u+1
    else:
      mageis[0]=np.concatenate([ mageis[0],time_grid0 ],axis=0) #time
      mageis[1]=np.concatenate([ mageis[1], e_ani0 ],axis=0) #e anisotropy
      mageis[2]=np.concatenate([ mageis[2], e_range0 ],axis=0) #energy level
    print(str(i))   

# delete all of those times when the energy range was incorrect 
  for h in range(0,25):      
    # first remove energy channels that changed over time  
    mageis_help1=mageis[1][:,h]   # e ani
    mageis_help2=mageis[2][:,h]   # energy
    mageis_help3=np.nanmedian(mageis_help2)
    help_index=np.nonzero(mageis_help2!=mageis_help3)[0]
    mageis[1][help_index,h]='nan' #load back updated array   
    mageis[2][help_index,h]='nan' #load back updated array   
    mageis_help1=mageis[1][:,h]   # e ani
    help_index=np.nonzero((mageis_help1<0) | (mageis_help1>100))[0]
    mageis[1][help_index,h]='nan' #load back updated array    
  # bad_mageis=np.argwhere(mageis[2][:,0]<0)  
  # mageis[0]=np.delete(mageis[0],bad_mageis)
  # mageis[1]=np.delete(mageis[1],bad_mageis,axis=0)
  # mageis[2]=np.delete(mageis[2],bad_mageis,axis=0)
  # #removing  bad energy bins
  # bad_mageis=np.argwhere(np.diff(mageis[2][:,0])>0)
  # mageis[0]=np.delete(mageis[0],bad_mageis)
  # mageis[1]=np.delete(mageis[1],bad_mageis,axis=0)
  # mageis[2]=np.delete(mageis[2],bad_mageis,axis=0)
# interpolate hope to vlf_time
 
  f_interp = interp1d( mageis[0], mageis[1].T,fill_value='nan',bounds_error=False)
  mageis_interp = f_interp(vlf_time).T
  
  mageis_interp[mageis_interp==0]='nan'
  
  #mageis_energy=mageis[2][0,:].T
  mageis_energy0=mageis[2].T
  mageis_energy=[]
  for r in range(25):
    mageis_energy=np.append(mageis_energy,np.nanmedian(mageis_energy0[r,:]))
    
  return mageis_interp, mageis_energy
     