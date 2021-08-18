# version 3: using sklearn's SOM package
# Goals:
# 1) using the prepared data by som_data_prep.py
# 2) train a SOM
# 3) plot trained results
# 4) identify nodes that correspond to the shadow spikes in the axial E-field data
# 5) remove those errenous lines from vlf, kp, dst and den_interp
# 6) use som v2 to train a new network on the cleaned data set

# %% train a new model

from os.path import dirname, join as pjoin
import scipy.io as sio
from scipy.io import readsav
import os
import numpy as np
## normalize data to ~2Hz frequency and drop some frequency bins above 2kHz
# vlf_norm=[]
# vlf_norm.append(np.log10(vlf[0][0:50,:])-np.log10(vlf[0][0,:]))
# vlf_norm.append(np.log10(vlf[1][0:50,:])-np.log10(vlf[1][0,:]))
# vlf_norm.append(np.log10(vlf[2][0:50,:])-np.log10(vlf[2][0,:]))

# new idea: focus on the 2 to 30 Hz range only
vlf_norm=[]
vlf_norm.append(np.log10(vlf[0][0:24,:])) #vlf0['b_spectra']
vlf_norm.append(np.log10(vlf[1][0:24,:])) #vlf0['e_spectra_spin']
vlf_norm.append(np.log10(vlf[2][0:24,:])) #vlf0['e_spectra_axial']

# %%
## train model

from sklearn_som.som import SOM
vlf_som=SOM(m=10,n=10,dim=np.shape(vlf_norm)[1],lr=1,sigma=1,max_iter=10000)
vlf_som.fit(vlf_norm[2].T)
predictions = vlf_som.predict(vlf_norm[2].T)


# %% use old model to remove shadow spikes

## save model
# import pickle
# filename=(r'D:\Research\Data\Van_Allen_Probes\Models\model_shadow.sav')
# pickle.dump(vlf_som, open(filename, 'wb'))

## load model
vlf_som = pickle.load(open(filename, 'rb'))
predictions = vlf_som2.predict(vlf_norm[2].T)

shadow_node=np.array([0,1,2,10,11,12,20,21,22,30,31,40,50]) # list of node id with shadow spike

del_shadow=[]
for i in range(0,len(predictions)):
  if np.sum(predictions[i]==shadow_node)>0:
    del_shadow.append(i)  

del_shadow=np.array(del_shadow)
## delete shadow spike intervals from all data products

vlf[0]=np.delete(vlf[0],del_shadow,axis=1)
vlf[1]=np.delete(vlf[1],del_shadow,axis=1)
vlf[2]=np.delete(vlf[2],del_shadow,axis=1)
vlf[3]=np.delete(vlf[3],del_shadow,axis=1)
vlf[4]=np.delete(vlf[4],del_shadow,axis=0)
vlf[5]=np.delete(vlf[5],del_shadow,axis=0)
vlf[7]=np.delete(vlf[7],del_shadow,axis=0)
vlf[8]=np.delete(vlf[8],del_shadow,axis=0)
vlf[9]=np.delete(vlf[9],del_shadow,axis=0)

den_interp=np.delete(den_interp,del_shadow,axis=0)
kp_interp=np.delete(kp_interp,del_shadow,axis=0)
dst_interp=np.delete(dst_interp,del_shadow,axis=0)

# %% new layout, with KP and DST indeces
#### load in frequency axis
data=readsav(r'D:\Research\Data\Van_Allen_Probes\rbsp_EMFISIS_freq_and_bandwidth.sav')
freq=np.reshape(data['freqs'][0:24],(-1,1)).T

import matplotlib.pyplot as plt
axial_grid=[]
node_size=[]
for i in range(0,100):    
    
    fig, axs = plt.subplots(2, 2)
    
    Y=np.mean(vlf_norm[0][:,np.argwhere((predictions==i))].T,axis=1)
    err=np.std(vlf_norm[0][:,np.argwhere((predictions==i))].T,axis=1)
    axs[0, 0].errorbar(np.squeeze((freq)), np.squeeze(Y), yerr=np.squeeze(err))
    axs[0, 0].set_title('Avg B-spectra, node: ' + str(i) + ', # of input data: ' + str(len(np.argwhere((predictions==i) ))))
    axs[0, 0].set_xscale('log')  
    axs[0, 0].set_xlabel('[Hz]')
    axs[0, 0].set_ylabel('[(nT)^2/Hz]')
    #axs[0,0].set_ylim( (0,-6))
    
    Y=np.mean(vlf_norm[2][:,np.argwhere((predictions==i))].T,axis=1)
    err=np.std(vlf_norm[2][:,np.argwhere((predictions==i))].T,axis=1)
    axs[0, 1].errorbar(np.squeeze((freq)), np.squeeze(Y), yerr=np.squeeze(err))
    axs[0, 1].set_title('Avg E-spectra axial')
    axs[0, 1].set_xscale('log')  
    axs[0, 1].set_xlabel('[nT]')
    axs[0, 1].set_ylabel('[(V/m)^2/Hz]')
    axs[0, 1].set_ylim( (-15,-2) )
    axial_grid.append(Y)
    node_size.append(len(np.argwhere((predictions==i))))

    axs[1,0].hist(kp_interp[np.argwhere((predictions==i))]/10,edgecolor='black',bins=np.arange(0,10))
    kp_node=kp_interp[np.argwhere((predictions==i))]/10
    axs[1, 0].set_title('KP index, % of KP>=4 cases: ' + str(100*(len(np.argwhere(kp_node>=4))/len(kp_node ))  )[0:4] )
    axs[1, 0].set_xlabel('[Hz]')
    axs[1, 0].set_ylabel('# of occurrence')
   
    xx=vlf[9][np.argwhere((predictions==i))]
    xx = xx[~np.isnan(xx)]
    yy=vlf[7][np.argwhere((predictions==i))]
    yy = yy[~np.isnan(yy)]
    
    # Polar histogram
    rbins = np.linspace(0, 7,7)
    abins = np.linspace(-np.pi, np.pi, 16)

    hist, _, _ = np.histogram2d(180*((yy-12)/12)*np.pi/180, xx, bins=(abins, rbins))
    A, R = np.meshgrid(abins, rbins)
    
    axs[1,1] = plt.subplot2grid((2,2), (1,1), polar=True)
    axs[1,1].pcolormesh(A, R, (hist.T/np.sum(hist)), cmap="viridis") #vmin=-6, vmax=-2
    axs[1,1].grid(True)
    
    fig.tight_layout()
    plt.show()

## attempt to localize shaadow spike nodes
axial_grid=np.array(axial_grid)
axial_grid=np.squeeze(axial_grid,axis=1)

node_size=np.array(node_size)

node_size[np.argwhere(axial_grid[:,0]>-6.5)] # selecting nodes with shadow spikes
np.sum(node_size[np.argwhere(axial_grid[:,0]>-6.5)])/np.sum(node_size)

#plt.scatter(axial_grid[:,0],axial_grid[:,3])
#plt.xlim((-8,-6))
# %% transform E-field data into B-field
m=1.67272*10**-27 # proton mass
m_e=9.109383*10**-31
q=1.60217662*10**-19 # charge
kb=1.38064852*10**-23 # boltzmann constant
c=299792458 # speed of light
eps0= 8.854187*10**-12

data=readsav(r'D:\Research\Data\Van_Allen_Probes\rbsp_EMFISIS_freq_and_bandwidth.sav')
freq=np.reshape(data['freqs'],(-1,1)).T
## plasma frequency
f_pe=np.sqrt(den_interp*100**3*q**2/(m_e*eps0))/(2*np.pi)
#f_pe=8980*np.sqrt(den_interp)

## electron cyclotron frequency
f_ce=(q*(vlf[4]*10**-9)/m_e)/(2*np.pi)

B=[]
for i in range(0,np.shape(vlf[0])[1]):
  E_square=vlf[1][:,i]*freq 
  B.append(np.square( (E_square/c**2)*(1-((f_pe[i]**2)/(freq*(freq-f_ce[i]))))))

B=np.squeeze(np.array(B),axis=1)