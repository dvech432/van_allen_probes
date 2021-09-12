#### load in FESA data, match it with vlf data
# Goal: load in those FESA files that are in the vlf_time
# Create an array that matches the size of vlf_time
# Fill up the array
####

def get_fesa(vlf_time):
  from os.path import dirname, join as pjoin
  import scipy.io as sio
  from scipy.io import readsav
  import os
  import numpy as np
  import pandas as pd
  import time
  import datetime
  
### check content of the folder
  foldername = r'D:\Research\Data\Van_Allen_Probes\Merged_plasma_mat\\'
  folder_content=[f for f in os.listdir(foldername) if not f.startswith('.')]
  folder_content.sort()

### load in FESA files
  fesa=[]
  u=0
  for i in range(0,len(folder_content)):
    mat_contents = sio.loadmat(foldername + folder_content[i]) # load in mat
    datenums = np.squeeze(np.array(mat_contents['FESA'][0][17]),axis=1) # convert mat time to unix time
    timestamps = pd.to_datetime(datenums-719529, unit='D')
    #index = pd.DatetimeIndex(timestamps)
    #time_grid0=np.array(index.astype(np.int64))/10**9
    unixtime=[]
    for p in range(0,len(timestamps)):
      unixtime.append(time.mktime(timestamps[p].timetuple()))
    time_grid0=np.array(unixtime)
    
    overlap=np.argwhere( (vlf_time>np.min(time_grid0))  & (vlf_time<np.max(time_grid0) ) )
    #print(str(len(overlap)))
    
    if len(overlap)>0:
      mat_contents['FESA'][0][18][mat_contents['FESA'][0][18]<0]=np.nan #replace missing values with nans
      bad_qual=np.argwhere(mat_contents['FESA'][0][6]>0)[:,0]
      mat_contents['FESA'][0][18][bad_qual,:]=np.nan #replace bad fit quality values with nans
      if u==0:
        fesa.append(time_grid0) #time
        fesa.append(mat_contents['FESA'][0][18]) #phase space density
        fesa.append(mat_contents['FESA'][0][19]) #energy level
        #fesa.append(mat_contents['FESA'][0][6]) #fesa fit quality
        u=u+1
      else:
        fesa[0]=np.concatenate([ fesa[0],time_grid0 ],axis=0) #time
        fesa[1]=np.concatenate([ fesa[1],mat_contents['FESA'][0][18] ],axis=0) #phase space density
        fesa[2]=np.concatenate([ fesa[2],mat_contents['FESA'][0][19] ],axis=1) #energy level
        #fesa[3]=np.concatenate([ fesa[3],mat_contents['FESA'][0][6] ],axis=0) #fesa fit quality
   
  #find the closest fesa to each power spectra
  #from datetime import datetime
  #start_time = datetime.now()
  prox_grid=[]  
  for i in range(0,len(vlf_time)):
    prox_grid.append([np.min(np.abs(vlf_time[i]-fesa[0])), np.argmin(np.abs(vlf_time[i]-fesa[0]))])
  #end_time = datetime.now()
  #print('Duration: {}'.format(end_time - start_time))

  # column 1: time delay between a given power spectra and the closest FESA
  # column 2: index of FESA in the vlf array
  prox_grid=np.array(prox_grid)  

  # generate an empty grid with size of vlf_time x number of energy bins in FESA
  vlf_fesa=np.empty( (np.shape(vlf_time)[0],  np.shape(fesa[2])[0]))
  #vlf_fesa=np.empty(    ( 900000,  np.shape(fesa[2])[0]))
  
  qq=prox_grid[:,1].astype(int)
  vlf_fesa=fesa[1][qq,:] # filled up array

  #  # pick all of those FESA that was measured +/- 100 sec wrt a power spectr
  fesa_ind=np.argwhere(prox_grid[:,0]>100) # all of those power spectra that does not have a corresponding FESA
  vlf_fesa[fesa_ind,:]=np.nan # replace those rows with nans
  
  return vlf_fesa


# test plot
# import matplotlib.pyplot as plt
# ind=np.random.randint(0,10000,5)
# for i in range(0,len(ind)):
#   plt.plot((fesa[2][:,0]),(fesa[1][ind[i],:]))
#   plt.xscale('log')
#   plt.yscale('log')
#   plt.ylabel('Differential electron flux')
#   plt.xlabel('Energy [eV]')

##########################
##########################
##########################

# k=30
# # DATE TIME to UNIX
# import time
# import datetime
# unixtime = time.mktime(timestamps[k].timetuple())

# # UNIX to DATE
# from datetime import datetime
# dt_object = datetime.fromtimestamp(unixtime)
# print('input:' + str(timestamps[k]))
# print('output:' + str(dt_object))


# overlap=np.argwhere( (vlf_time>np.min(fesa[0]))  & (vlf_time<np.max(fesa[0]) ) )

# plt.hist(fesa[0],density=True)
# plt.hist(vlf_time,density=True)

# plt.plot(fesa[0])
# plt.plot(vlf_time)