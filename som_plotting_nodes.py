### load in all data 

# %% step 1

from os.path import dirname, join as pjoin
import scipy.io as sio
from scipy.io import readsav
import os
import numpy as np
import pandas as pd
### checking primary folder with vlf data
foldername = r'E:\rbsp_hiss_hsr_daily_vlf_files\a\\'
folder_content=[f for f in os.listdir(foldername) if not f.startswith('.')]
folder_content.sort()
from del_bad_data import del_bad_data #load in the function that removes bad data

###### load in 250 randomly selected days from the range of 0:1400
# import random
# ind=random.sample(range(1400), 250)
# ind.sort()
# ind=np.array(ind)

# 1 )load in the daily VLF files
print('Step 1 started')
#np.savetxt(r'D:\Research\Data\Van_Allen_Probes\ind.txt',ind)
filename = r'D:\Research\Data\Van_Allen_Probes\ind.txt'
ind = (np.genfromtxt(filename, dtype=float, invalid_raise=False, missing_values='', usemask=False, filling_values=0.0, skip_header=0))
ind2=[]
for i in range(0,len(ind)):
  ind2.append(int(ind[i]))    
ind=np.array(ind2)

for i in range(0,len(ind)): # 1) load in the daily ULF files
#for i in range(0,100): # 1) load in the daily ULF files
  #vlf0=readsav(foldername + folder_content[i]) # old method, 0-->i
  vlf0=readsav(foldername + folder_content[ind[i]]) # load in randomly selected data
  if i==0:
    vlf=del_bad_data(vlf0) # 2) drop those lines where the data quality flag is not 0
  else:
    vlf0=del_bad_data(vlf0)
    vlf[0]=np.concatenate([ vlf[0],vlf0[0] ],axis=1) #vlf0['b_spectra']
    vlf[1]=np.concatenate([ vlf[1],vlf0[1] ],axis=1) #vlf0['e_spectra_spin']
    vlf[2]=np.concatenate([ vlf[2],vlf0[2] ],axis=1) #vlf0['e_spectra_axial']
    vlf[3]=np.concatenate([ vlf[3],vlf0[3] ],axis=1) #vlf0['compressability']
    
    vlf[4]=np.concatenate([ vlf[4],vlf0[4] ],axis=0) #vlf0['b_mag']
    vlf[5]=np.concatenate([ vlf[5],vlf0[5] ],axis=0) #vlf0['b_time']
    
    vlf[6]=np.concatenate([ vlf[6],vlf0[6] ],axis=0) #vlf0['unixtime']
    vlf[7]=np.concatenate([ vlf[7],vlf0[7] ],axis=0) #vlf0['mlt']
    vlf[8]=np.concatenate([ vlf[8],vlf0[8] ],axis=0) #vlf0['mlat']
    vlf[9]=np.concatenate([ vlf[9],vlf0[9] ],axis=0) #vlf0['lshell']
  print(i)  
# 2) drop those lines where the data quality flag is not 0    
print('Step 2 started')
from scipy.interpolate import interp1d #interpolate vlf[7]-[9] to the time stamps at vlf[5]
f_interp = interp1d( vlf[6],  vlf[7],fill_value='nan',bounds_error=False)
vlf[7] = f_interp(vlf[5])
f_interp = interp1d( vlf[6],  vlf[8],fill_value='nan',bounds_error=False)
vlf[8] = f_interp(vlf[5])
f_interp = interp1d( vlf[6],  vlf[9],fill_value='nan',bounds_error=False)
vlf[9] = f_interp(vlf[5])    
 
# 3) load in density data  
print('Step 3 started')
foldername = r'E:\data_TS04D_all_L\a\\'
#folder_content=[f for f in os.listdir(foldername) if not f.startswith('.')]
folder_content=[f for f in os.listdir(foldername) if not f.startswith('.') | f.endswith('g')]
folder_content.sort()

from process_den import process_den #load in the function to process density

for i in range(0,len(folder_content)):
 #if folder_content[i][-3:]=='sav': # read in only sav files, ignore png extension
  den_str0=readsav(foldername + folder_content[i])
  if i==0:
    den_str=process_den(den_str0)    
  else:
    den_str0=process_den(den_str0)
    den_str[0]=np.concatenate([ den_str[0], den_str0[0] ],axis=0)
    den_str[1]=np.concatenate([ den_str[1], den_str0[1] ],axis=0)  
      
# 4) interpolate density to the VLF files     #den_df = pd.DataFrame({"Density": den_str[1]},index=den_str[0] )
print('Step 4 started')
f_interp = interp1d( den_str[0],  den_str[1],fill_value='nan',bounds_error=False)
den_interp = f_interp(vlf[5])

# 5) drop those data points where density is below 50 cm^-3 or no data is available
print('Step 5 started')
low_den=np.argwhere( (den_interp<50) |  np.isnan(den_interp)==True)
vlf[0]=np.delete(vlf[0],low_den,axis=1)
vlf[1]=np.delete(vlf[1],low_den,axis=1)
vlf[2]=np.delete(vlf[2],low_den,axis=1)
vlf[3]=np.delete(vlf[3],low_den,axis=1)
vlf[4]=np.delete(vlf[4],low_den,axis=0)
vlf[5]=np.delete(vlf[5],low_den,axis=0)
vlf[7]=np.delete(vlf[7],low_den,axis=0)
vlf[8]=np.delete(vlf[8],low_den,axis=0)
vlf[9]=np.delete(vlf[9],low_den,axis=0)

den_interp=np.delete(den_interp,low_den,axis=0)

# 6) load in KP, DST index data and interpolate to VLF data
print('Step 6 started')
filename = r'D:\Research\Data\Van_Allen_Probes\KP.mat'
kp = sio.loadmat(filename)
kp=kp['KP']
f_interp = interp1d( kp[:,0],  kp[:,1],fill_value='nan',bounds_error=False)
kp_interp = f_interp(vlf[5])
f_interp = interp1d( kp[:,0],  kp[:,2],fill_value='nan',bounds_error=False)
dst_interp = f_interp(vlf[5])

# 7) remove shadow spikes
print('Step 7 started')
from shadow_removal import shadow_removal
vlf, den_interp, kp_interp, dst_interp = shadow_removal(vlf, den_interp, kp_interp, dst_interp)

# 8) do SOM on the cleaned data set
print('Step 8 started')
from spin_som import spin_som
predictions, vlf_norm = spin_som(vlf)

# delete compressibility since it is useless anyway
vlf[3]=[]

# %% save and load file

from scipy.io import savemat
filename=(r'D:\Research\Data\Van_Allen_Probes\vlf.mat')
savemat(filename, {"foo":vlf})

# %% categories based on empirical determination

no_hiss=[0, 1, 2, 3,9,10,11,18,19,20,28,29,30,40,41,42,50,51,52,60,61,62,70,71,72,80,81,82,85,86,90,91,92,93,95,96]
low_freq_hiss=[16,17,26,27,36,37,38,39,47,48,49,57,58,59,67,68,69,77,78,79,87,88,89,97,98,99]
regular_hiss=[4,5,6,7,8,12,13,14,15,21,22,23,24,25,31,32,33,34,35,43,44,45,46,53,54,55,56,63,64,65,66,73,74,75,76,83,84,94]



# %% calculating intergral of power spectra between 20-100 Hz / 20-3000 Hz

data=readsav(r'D:\Research\Data\Van_Allen_Probes\rbsp_hiss_hsr_a_Espin_noise.sav')
n_floor=np.reshape(data['espin_noise_floor_sigmas'],(-1,1)).T # use 1 sigma above noise floor

data=readsav(r'D:\Research\Data\Van_Allen_Probes\rbsp_hiss_hsr_a_scm_noise_low_gain.sav')
scm_floor=np.reshape(data['scm_noise_floor_levels']+data['scm_noise_floor_sigmas'],(-1,1)).T # use 1 sigma above noise floor

data=readsav(r'D:\Research\Data\Van_Allen_Probes\rbsp_EMFISIS_freq_and_bandwidth.sav')
freq=np.reshape(data['freqs'][0:50],(-1,1)).T
freq_long=np.reshape(data['freqs'][0:65],(-1,1)).T

## transform E-field data into B-field
m=1.67272*10**-27 # proton mass
m_e=9.109383*10**-31
q=1.60217662*10**-19 # charge
kb=1.38064852*10**-23 # boltzmann constant
c=299792458 # speed of light
eps0= 8.854187*10**-12

freq2=np.squeeze((freq))

psd_ratio=[]

import matplotlib.pyplot as plt
for i in range(0,100): # go through all nodes

  E_spin=vlf[1][:,np.argwhere((predictions==i))] 
  E_spin=np.squeeze(E_spin,axis=2)
    #### replace those frequency bins that are below the noise floor
  for j in range(0,np.shape(n_floor)[1]):
    below_floor=np.argwhere(np.log10(E_spin[j,:])<n_floor[0,j])
    E_spin[j,below_floor]=np.nan
     
    ### add the spin and axial data 
  Y=np.nanmean(np.log10(E_spin+np.squeeze(vlf[2][:,np.argwhere((predictions==i))],axis=2)),axis=1)    
  Y=Y[0:50]
  
  # integration part
  low_int=np.trapz(Y[8:24], x=np.log10(freq2[8:24]))
  tot_int=np.trapz(Y[24:44], x=np.log10(freq2[24:44]))
  psd_ratio.append(low_int/tot_int)
  print(i)  
psd_ratio=np.array(psd_ratio)  


# %% plotting polar histograms


i=99
#POLAR STUFF
xx=vlf[9][np.argwhere((predictions==i))]
xx = xx[~np.isnan(xx)]
yy=vlf[7][np.argwhere((predictions==i))]
yy = yy[~np.isnan(yy)]
    
# # Polar histogram
rbins = np.linspace(0, 10,11)
abins = np.linspace(-np.pi, np.pi, 36)

hist, _, _ = np.histogram2d(180*((yy-12)/12)*np.pi/180, xx, bins=(abins, rbins))
A, R = np.meshgrid(abins, rbins)
   
plt.figure(dpi=750) 
plt.subplot2grid((2,2), (1,1), polar=True)
plt.pcolormesh(A, R, (hist.T/np.sum(hist)), cmap="viridis") #vmin=-6, vmax=-2
plt.grid(True)
plt.colorbar()   
 
plt.tight_layout()
plt.show()


# %% polar histogram incorporating multiple nodes





# %% checking solar wind params and their correlation with low frequency hiss

from get_omni import get_omni
omni_interp=get_omni(vlf[5])


import matplotlib.pyplot as plt

v_moving_avg=pd.Series(omni_interp[:,5]).rolling(24).mean()
dv=np.array(omni_interp[:,5]-v_moving_avg) # V_sw - 2 hr rolling mean


for i in range(0,100):    
    
    fig, axs = plt.subplots(2, 2)
    
    #### PANEL 1
    axs[0, 0].hist(omni_interp[np.argwhere((predictions==i)),4],edgecolor='black',bins=10 )
    axs[0, 0].set_xlabel('$B_Z [nT]$')
    axs[0, 0].set_ylabel('# of occurrence') 
    axs[0, 0].grid(True)
    axs[0, 0].set_xlim((-20,20))
    #### PANEL 2
    axs[0, 1].hist(omni_interp[np.argwhere((predictions==i)),5],edgecolor='black',bins=10 )
    axs[0, 1].set_xlabel('$V_{sw} [km/s]$')
    axs[0, 1].set_ylabel('# of occurrence') 
    axs[0, 1].grid(True)
    axs[0, 1].set_xlim((250,800))
    
    #### PANEL 3
    axs[1, 0].hist(dv[np.argwhere((predictions==i))],edgecolor='black',bins=40 )
    axs[1, 0].set_xlabel('$V_{sw} [km/s]$')
    axs[1, 0].set_ylabel('# of occurrence') 
    axs[1, 0].grid(True)
    axs[1, 0].set_xlim((-50,50))
    
    #### PANEL 4
    axs[1, 1].hist(omni_interp[np.argwhere((predictions==i)),6],edgecolor='black',bins=30 )
    axs[1, 1].set_xlabel('Solar wind dynamic pressure [nPa]')
    axs[1, 1].set_ylabel('# of occurrence') 
    axs[1, 1].grid(True)
    axs[1, 1].set_xlim((0,20))
    #axs[1, 1].set_xscale('log')
        
    fig.tight_layout()
    plt.show()
    
    filename = r'D:\Research\Data\Van_Allen_Probes\SOM_sw_params\\' + str(i).zfill(3) + '_QQ.jpg'
    fig.savefig(filename,bbox_inches='tight')
   
    plt.clf()
    fig.clear('all')
    plt.close(fig)
    plt.close()
    
    
# %% compare all three categories 

no_hiss=[0, 1, 2, 3,9,10,11,18,19,20,28,29,30,40,41,42,50,51,52,60,61,62,70,71,72,80,81,82,85,86,90,91,92,93,95,96]
low_freq_hiss=[16,17,26,27,36,37,38,39,47,48,49,57,58,59,67,68,69,77,78,79,87,88,89,97,98,99]
regular_hiss=[4,5,6,7,8,12,13,14,15,21,22,23,24,25,31,32,33,34,35,43,44,45,46,53,54,55,56,63,64,65,66,73,74,75,76,83,84,94]
   

from get_omni import get_omni
omni_interp=get_omni(vlf[5])


import matplotlib.pyplot as plt

v_moving_avg=pd.Series(omni_interp[:,5]).rolling(24).mean()
dv=np.array(omni_interp[:,5]-v_moving_avg) # V_sw - 2 hr rolling mean


p1=[]
p2=[]
p3=[]
p4=[]
for i in range(0,len(no_hiss)):    
    p1.append(omni_interp[np.argwhere((predictions==no_hiss[i])),4])
    #### PANEL 2
    p2.append(omni_interp[np.argwhere((predictions==no_hiss[i])),5])  
    #### PANEL 3
    p3.append(dv[np.argwhere((predictions==no_hiss[i]))])
    #### PANEL 4
    p4.append(omni_interp[np.argwhere((predictions==no_hiss[i])),6])

p1=np.vstack(p1)
p2=np.vstack(p2)
p3=np.vstack(p3)
p4=np.vstack(p4)

q1=[]
q2=[]
q3=[]
q4=[]
for i in range(0,len(low_freq_hiss)):    
    q1.append(omni_interp[np.argwhere((predictions==low_freq_hiss[i])),4])
    #### PANEL 2
    q2.append(omni_interp[np.argwhere((predictions==low_freq_hiss[i])),5])  
    #### PANEL 3
    q3.append(dv[np.argwhere((predictions==low_freq_hiss[i]))])
    #### PANEL 4
    q4.append(omni_interp[np.argwhere((predictions==low_freq_hiss[i])),6])

q1=np.vstack(q1)
q2=np.vstack(q2)
q3=np.vstack(q3)
q4=np.vstack(q4)

e1=[]
e2=[]
e3=[]
e4=[]
for i in range(0,len(regular_hiss)):    
    e1.append(omni_interp[np.argwhere((predictions==regular_hiss[i])),4])
    #### PANEL 2
    e2.append(omni_interp[np.argwhere((predictions==regular_hiss[i])),5])  
    #### PANEL 3
    e3.append(dv[np.argwhere((predictions==regular_hiss[i]))])
    #### PANEL 4
    e4.append(omni_interp[np.argwhere((predictions==regular_hiss[i])),6])    

e1=np.vstack(e1)
e2=np.vstack(e2)
e3=np.vstack(e3)
e4=np.vstack(e4)    

###### final histogram

fig, axs = plt.subplots(2, 2)
    
    #### PANEL 1
axs[0, 0].hist(p1,edgecolor='black',bins=20, density=True, alpha=0.5 )
axs[0, 0].hist(q1,edgecolor='black',bins=20, density=True, alpha=0.5 )
axs[0, 0].hist(e1,edgecolor='black',bins=20, density=True, alpha=0.5 )
axs[0, 0].set_xlabel('$B_Z [nT]$')
axs[0, 0].set_ylabel('# of occurrence') 
axs[0, 0].grid(True)
axs[0, 0].set_xlim((-10,10))
    #### PANEL 2
axs[0, 1].hist(p2,edgecolor='black',bins=20, density=True, alpha=0.5 )
axs[0, 1].hist(q2,edgecolor='black',bins=20, density=True, alpha=0.5 )
axs[0, 1].hist(e2,edgecolor='black',bins=20, density=True, alpha=0.5 )
axs[0, 1].set_xlabel('$V_{sw} [km/s]$')
axs[0, 1].set_ylabel('# of occurrence') 
axs[0, 1].grid(True)
axs[0, 1].set_xlim((250,800))
    
    #### PANEL 3
axs[1, 0].hist(p3,edgecolor='black',bins=50, density=True, alpha=0.5 )
axs[1, 0].hist(q3,edgecolor='black',bins=50, density=True, alpha=0.5 )
axs[1, 0].hist(e3,edgecolor='black',bins=50, density=True, alpha=0.5 )
axs[1, 0].set_xlabel('$V_{sw} [km/s]$')
axs[1, 0].set_ylabel('# of occurrence') 
axs[1, 0].grid(True)
axs[1, 0].set_xlim((-50,50))
    
    #### PANEL 4
axs[1, 1].hist(p4,edgecolor='black',bins=50, density=True, alpha=0.5 )
axs[1, 1].hist(q4,edgecolor='black',bins=50, density=True, alpha=0.5 )
axs[1, 1].hist(e4,edgecolor='black',bins=50, density=True, alpha=0.5 )
axs[1, 1].set_xlabel('Solar wind dynamic pressure [nPa]')
axs[1, 1].set_ylabel('# of occurrence') 
axs[1, 1].grid(True)
axs[1, 1].set_xlim((0,20))
    #axs[1, 1].set_xscale('log')
        
fig.tight_layout()
plt.show()

# %% showing the three main categories in spatial domain

# first collect pred IDs for each three groups


i=99
#POLAR STUFF
xx=vlf[9][np.argwhere((predictions==i))]
xx = xx[~np.isnan(xx)]
yy=vlf[7][np.argwhere((predictions==i))]
yy = yy[~np.isnan(yy)]
    
# # Polar histogram
rbins = np.linspace(0, 10,11)
abins = np.linspace(-np.pi, np.pi, 36)

hist, _, _ = np.histogram2d(180*((yy-12)/12)*np.pi/180, xx, bins=(abins, rbins))
A, R = np.meshgrid(abins, rbins)
   
plt.figure(dpi=750) 
plt.subplot2grid((2,2), (1,1), polar=True)
plt.pcolormesh(A, R, (hist.T/np.sum(hist)), cmap="viridis") #vmin=-6, vmax=-2
plt.grid(True)
plt.colorbar()   
 
plt.tight_layout()
plt.show()