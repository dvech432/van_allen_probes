# to do list:
# 1 )load in the daily VLF files
# 2) drop those lines where the data quality flag is not 0
# 3) load in the density structure files
# 4) interpolate it to the ULF files
# 5) Drop those ULF data where density is below 50 cm^-3

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

#np.savetxt(r'D:\Research\Data\Van_Allen_Probes\ind.txt',ind)
filename = r'D:\Research\Data\Van_Allen_Probes\ind.txt'
ind = (np.genfromtxt(filename, dtype=float, invalid_raise=False, missing_values='', usemask=False, filling_values=0.0, skip_header=0))
ind2=[]
for i in range(0,len(ind)):
  ind2.append(int(ind[i]))    
ind=np.array(ind2)

for i in range(0,len(ind)): # 1) load in the daily ULF files
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
    print(str(i))
  
    
from scipy.interpolate import interp1d #interpolate vlf[7]-[9] to the time stamps at vlf[5]
f_interp = interp1d( vlf[6],  vlf[7],fill_value='nan',bounds_error=False)
vlf[7] = f_interp(vlf[5])
f_interp = interp1d( vlf[6],  vlf[8],fill_value='nan',bounds_error=False)
vlf[8] = f_interp(vlf[5])
f_interp = interp1d( vlf[6],  vlf[9],fill_value='nan',bounds_error=False)
vlf[9] = f_interp(vlf[5])    
 
# 3) load in density data  
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
    print(str(i))
     
 
# 4) interpolate density to the VLF files     #den_df = pd.DataFrame({"Density": den_str[1]},index=den_str[0] )
f_interp = interp1d( den_str[0],  den_str[1],fill_value='nan',bounds_error=False)
den_interp = f_interp(vlf[5])

# 5) drop those data points where density is below 50 cm^-3 or no data is available

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
filename = r'D:\Research\Data\Van_Allen_Probes\KP.mat'
kp = sio.loadmat(filename)
kp=kp['KP']
f_interp = interp1d( kp[:,0],  kp[:,1],fill_value='nan',bounds_error=False)
kp_interp = f_interp(vlf[5])
f_interp = interp1d( kp[:,0],  kp[:,2],fill_value='nan',bounds_error=False)
dst_interp = f_interp(vlf[5])


# 7) remove shadow spikes
from shadow_removal import shadow_removal
vlf, den_interp, kp_interp, dst_interp = shadow_removal(vlf, den_interp, kp_interp, dst_interp)

# 8) do SOM on the cleaned data set
from spin_som import spin_som
predictions, vlf_norm = spin_som(vlf)


################## everything is ready for plotting

# %% make summary plots

# 1) normalized E-spin
# 2) non-normalized E-spin+E-axial
# 3) spin + axial converted to B-fiedl
# 4) spatial location

data=readsav(r'D:\Research\Data\Van_Allen_Probes\rbsp_hiss_hsr_a_Espin_noise.sav')
n_floor=np.reshape(data['espin_noise_floor_levels'],(-1,1)).T

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


import matplotlib.pyplot as plt
for i in range(0,100):    
    
    fig, axs = plt.subplots(2, 2)
    
    #### PANEL 1
    # Y=np.mean(vlf_norm[1][:,np.argwhere((predictions==i))].T,axis=1)
    # err=np.std(vlf_norm[1][:,np.argwhere((predictions==i))].T,axis=1)
    # axs[0, 0].errorbar(np.squeeze((freq)), np.squeeze(Y), yerr=np.squeeze(err))
    # axs[0, 0].set_title('Norm. Avg, E-spectra spin, id: ' + str(i) + ', #: ' + str(len(np.argwhere((predictions==i) ))))
    # axs[0, 0].set_xscale('log')  
    # axs[0, 0].set_xlabel('[nT]')
    # axs[0, 0].set_ylabel('[(V/m)^2/Hz]')
    axs[0,0].hist(kp_interp[np.argwhere((predictions==i))]/10,edgecolor='black',bins=np.arange(0,10))
    kp_node=kp_interp[np.argwhere((predictions==i))]/10
    axs[0, 0].set_title('id: ' + str(i) + ', #: ' + str(len(np.argwhere((predictions==i) ))) + ', % of KP>=4 cases: ' + str(100*(len(np.argwhere(kp_node>=4))/len(kp_node ))  )[0:4] )
    axs[0, 0].set_xlabel('KP index')
    axs[0, 0].set_ylabel('# of occurrence')
    
    #### PANEL 2
    #### take that spin axis data that belong to node i
    E_spin=vlf[1][:,np.argwhere((predictions==i))] 
    E_spin=np.squeeze(E_spin,axis=2)
    #### replace those frequency bins that are below the noise floor
    for j in range(0,np.shape(n_floor)[1]):
      below_floor=np.argwhere(np.log10(E_spin[j,:])<n_floor[0,j])
      E_spin[j,below_floor]=np.nan
     
    ### add the spin and axial data 
    Y=np.nanmean(np.log10(E_spin+np.squeeze(vlf[2][:,np.argwhere((predictions==i))],axis=2)),axis=1)
    err=np.nanstd(np.log10(E_spin+np.squeeze(vlf[2][:,np.argwhere((predictions==i))],axis=2)),axis=1)
    axs[0, 1].errorbar(np.squeeze((freq)), Y[0:50], yerr=err[0:50])
    axs[0, 1].set_title('Cleaned E spin + E axial')
    axs[0, 1].set_xscale('log')  
    axs[0, 1].set_xlabel('[Hz]')
    axs[0, 1].set_ylabel('[(V/m)^2/Hz]') 
    axs[0, 1].grid(True)
    axs[0,1].set_ylim( (-13,-6)) 
    
    #### PANEL 3
    ### Create the 2D grid of B-field values (derived from E-field), take the avg and std and plot it
    ## plasma frequency
    f_pe=np.sqrt(den_interp[np.argwhere((predictions==i))]*100**3*q**2/(m_e*eps0))/(2*np.pi)
    ## electron cyclotron frequency
    f_ce=(q*(vlf[4][np.argwhere((predictions==i))]*10**-9)/m_e)/(2*np.pi)
    freq_grid=[]
    for j in range(0,np.shape(freq)[1]):
     freq_grid.append(1-((f_pe**2)/(freq[0,j]*(freq[0,j]-f_ce))))
    freq_grid=np.array(freq_grid)
    freq_grid=np.squeeze(freq_grid,axis=2)
    ### last step: apply the whistler dispersion relation to convert E field to B field
    E_square=E_spin[0:50,:]+np.squeeze(vlf[2][0:50,np.argwhere((predictions==i))])
    B= np.sqrt( ((1/c**2)*E_square*freq.T) *freq_grid)*10**9
    
    
    
    Y=np.nanmean(np.log10( ((B**2))/freq.T),axis=1)
    err=np.nanstd(np.log10(B),axis=1)
    
    Y_real_B=np.mean(np.log10(vlf[0][0:50,np.argwhere((predictions==i))].T),axis=1)
    err_real_B=np.std(np.log10(vlf[0][0:50,np.argwhere((predictions==i))].T),axis=1)
    
    axs[1, 0].errorbar(np.squeeze((freq)), Y[0:50], yerr=err[0:50],c='red',label='From E-field')
    axs[1, 0].errorbar(np.squeeze((freq)), Y_real_B[0,0:50], yerr=err_real_B[0,0:50],c='black',label='SCM')
    axs[1, 0].set_title('Converted E spin + E axial to B')
    axs[1, 0].set_xscale('log')  
    axs[1, 0].set_xlabel('[Hz]')
    axs[1, 0].set_ylabel('[Log10 (nT)^2/Hz]') 
    axs[1, 0].grid(True)
    axs[1,0].legend(loc="upper right")
    
    #### PANEL 4    
    axs[1, 1].scatter(np.squeeze((freq)), (10**Y_real_B[0,0:50])/(10**Y[0:50]))
    axs[1, 1].set_title('Real B / Derived B')
    axs[1, 1].set_xscale('log')  
    axs[1, 1].set_xlabel('[Hz]')
    axs[1,1].set_yscale('log')
    axs[1, 1].set_ylabel(' ') 
    axs[1, 1].grid(True)
    axs[1,1].set_ylim( (0.1,10)) 
    
    
    
    
    # POLAR STUFF
    # xx=vlf[9][np.argwhere((predictions==i))]
    # xx = xx[~np.isnan(xx)]
    # yy=vlf[7][np.argwhere((predictions==i))]
    # yy = yy[~np.isnan(yy)]
    
    # # Polar histogram
    # rbins = np.linspace(0, 7,7)
    # abins = np.linspace(-np.pi, np.pi, 16)

    # hist, _, _ = np.histogram2d(180*((yy-12)/12)*np.pi/180, xx, bins=(abins, rbins))
    # A, R = np.meshgrid(abins, rbins)
    
    # axs[1,1] = plt.subplot2grid((2,2), (1,1), polar=True)
    # axs[1,1].pcolormesh(A, R, (hist.T/np.sum(hist)), cmap="viridis") #vmin=-6, vmax=-2
    # axs[1,1].grid(True)
    
    fig.tight_layout()
    plt.show()
    
    # filename = r'D:\Research\Data\Van_Allen_Probes\SOM_figures\\' + str(i).zfill(3) + '_QQ.jpg'
    # fig.savefig(filename,bbox_inches='tight')
   
    # plt.clf()
    # fig.clear('all')
    # plt.close(fig)
    # plt.close()


###### comments on the conversion synthax

## go from datetime to unix time
# from datetime import datetime
# dates = [datetime(2012, 5, 1), datetime(2012, 5, 2), datetime(2012, 5, 3)]
# index = pd.DatetimeIndex(dates)
# index.astype(np.int64)

# ## go from unix time to datetime
#from datetime import datetime
#datetime.fromtimestamp(kp[60001,0])
#datetime.fromtimestamp(vlf[4][1])

#NN = 719529 # NN = datenum('01-jan-1970 00:00:00','dd-mmm-yyyy HH:MM:SS')    
#b_time = (vlf[4]/86400) + NN # old
#b_time2 = pd.to_datetime(b_time-719529, unit='D') # old
# %% save output matrix
#from scipy.io import savemat
#savemat(r'D:\Research\Data\Van_Allen_Probes\SOM\rbsp.mat', mdict={'arr': vlf[0]})
