# version 2: using sklearn's SOM package

#### using the prepared data by som_data_prep.py
### 1) train a SOM
### 2) plot trained results

from os.path import dirname, join as pjoin
import scipy.io as sio
from scipy.io import readsav
import os
import numpy as np
## normalize data to ~2Hz frequency and drop some frequency bins above 2kHz
vlf_norm=[]
vlf_norm.append(np.log10(vlf[0][0:50,:])-np.log10(vlf[0][0,:]))
vlf_norm.append(np.log10(vlf[1][0:50,:])-np.log10(vlf[1][0,:]))
vlf_norm.append(np.log10(vlf[2][0:50,:])-np.log10(vlf[2][0,:]))

# new idea: focus on the 2 to 30 Hz range only
# vlf_norm=[]
# vlf_norm.append(np.log10(vlf[0][0:20,:])) #vlf0['b_spectra']
# vlf_norm.append(np.log10(vlf[1][0:20,:])) #vlf0['e_spectra_spin']
# vlf_norm.append(np.log10(vlf[2][0:20,:])) #vlf0['e_spectra_axial']
## train model
### sklearn version of SOM

from sklearn_som.som import SOM
vlf_som=SOM(m=10,n=10,dim=np.shape(vlf_norm)[1],lr=1,sigma=1,max_iter=10000)
vlf_som.fit(vlf_norm[1].T)
predictions = vlf_som.predict(vlf_norm[1].T)

# %% save model

import pickle
filename=(r'D:\Research\Data\Van_Allen_Probes\Models\model_spin.sav')
pickle.dump(vlf_som, open(filename, 'wb'))

# %% load model
filename=(r'D:\Research\Data\Van_Allen_Probes\Models\model_spin.sav')
vlf_som = pickle.load(open(filename, 'rb'))
predictions = vlf_som2.predict(vlf_norm[1].T)
# %%
## plot distribution of input vectors among the nodes
# import matplotlib.pyplot as plt
# k=0;
# w_x=[]
# w_y=[]
# for i in range(0,10): # column
#   for j in range(0,10):    # row
#        for u in range(0,len(np.argwhere(predictions==k))):
#          w_x.append(i)
#          w_y.append(j)
#        k=k+1

# w_x = np.array(w_x)
# w_y = np.array(w_y)

# plt.hist2d(w_x+.5+(np.random.rand(len(w_x))-.5)*.8, w_y+.5+(np.random.rand(len(w_x))-.5)*.8, bins=10, cmap='Blues')
# cb = plt.colorbar(label='count in bin')
# plt.show()

# %% new layout, with KP and DST indeces
#### load in frequency axis
data=readsav(r'D:\Research\Data\Van_Allen_Probes\rbsp_EMFISIS_freq_and_bandwidth.sav')
freq=np.reshape(data['freqs'][0:50],(-1,1)).T

import matplotlib.pyplot as plt
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
    
    Y=np.mean(vlf_norm[1][:,np.argwhere((predictions==i))].T,axis=1)
    err=np.std(vlf_norm[1][:,np.argwhere((predictions==i))].T,axis=1)
    axs[0, 1].errorbar(np.squeeze((freq)), np.squeeze(Y), yerr=np.squeeze(err))
    axs[0, 1].set_title('Avg E-spectra spin')
    axs[0, 1].set_xscale('log')  
    axs[0, 1].set_xlabel('[nT]')
    axs[0, 1].set_ylabel('[(V/m)^2/Hz]')

    axs[1,0].hist(kp_interp[np.argwhere((predictions==i))]/10,edgecolor='black',bins=np.arange(0,10))
    kp_node=kp_interp[np.argwhere((predictions==i))]/10
    axs[1, 0].set_title('KP index, % of KP>=4 cases: ' + str(100*(len(np.argwhere(kp_node>=4))/len(kp_node ))  )[0:4] )
    axs[1, 0].set_xlabel('[Hz]')
    axs[1, 0].set_ylabel('# of occurrence')
   
    xx=vlf[9][np.argwhere((predictions==i))]
    xx = xx[~np.isnan(xx)]
    yy=vlf[7][np.argwhere((predictions==i))]
    yy = yy[~np.isnan(yy)]
    # axs[1, 1].hexbin(xx,yy,gridsize=20, bins='log', cmap='viridis' )
    # axs[1, 1].set_title('L-shell vs. Mlt')
    # axs[1, 1].set_xlabel('L-shell')
    # axs[1, 1].set_ylabel('Mlt')
    
    # Polar histogram
    rbins = np.linspace(0, 7,7)
    abins = np.linspace(-np.pi, np.pi, 16)

    hist, _, _ = np.histogram2d(180*((yy-12)/12)*np.pi/180, xx, bins=(abins, rbins))
    A, R = np.meshgrid(abins, rbins)
    
    axs[1,1] = plt.subplot2grid((2,2), (1,1), polar=True)
    axs[1,1].pcolormesh(A, R, (hist.T/np.sum(hist)), cmap="viridis") #vmin=-6, vmax=-2
    #cbar = plt.colorbar(pc)
    #cbar.set_label('Log10 probability')
    axs[1,1].grid(True)
    
    fig.tight_layout()
    plt.show()


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
    B= np.sqrt( ((E_spin[0:50,:]+np.squeeze(vlf[2][0:50,np.argwhere((predictions==i))]))/(c**2) )*freq_grid)
    
    Y=np.nanmean(np.log10(B),axis=1)
    err=np.nanstd(np.log10(B),axis=1)
    axs[1, 0].errorbar(np.squeeze((freq)), Y[0:50], yerr=err[0:50])
    axs[1, 0].set_title('Converted E spin + E axial to B')
    axs[1, 0].set_xscale('log')  
    axs[1, 0].set_xlabel('[Hz]')
    axs[1, 0].set_ylabel('[Log10 nT]') 
    axs[1, 0].grid(True)
    #### PANEL 4    
    
    xx=vlf[9][np.argwhere((predictions==i))]
    xx = xx[~np.isnan(xx)]
    yy=vlf[7][np.argwhere((predictions==i))]
    yy = yy[~np.isnan(yy)]
    # axs[1, 1].hexbin(xx,yy,gridsize=20, bins='log', cmap='viridis' )
    # axs[1, 1].set_title('L-shell vs. Mlt')
    # axs[1, 1].set_xlabel('L-shell')
    # axs[1, 1].set_ylabel('Mlt')
    
    # Polar histogram
    rbins = np.linspace(0, 7,7)
    abins = np.linspace(-np.pi, np.pi, 16)

    hist, _, _ = np.histogram2d(180*((yy-12)/12)*np.pi/180, xx, bins=(abins, rbins))
    A, R = np.meshgrid(abins, rbins)
    
    axs[1,1] = plt.subplot2grid((2,2), (1,1), polar=True)
    axs[1,1].pcolormesh(A, R, (hist.T/np.sum(hist)), cmap="viridis") #vmin=-6, vmax=-2
    #cbar = plt.colorbar(pc)
    #cbar.set_label('Log10 probability')
    axs[1,1].grid(True)
    
    fig.tight_layout()
    plt.show()
    
    filename = r'D:\Research\Data\Van_Allen_Probes\SOM_figures\\' + str(i).zfill(3) + '_QQ.jpg'
    fig.savefig(filename,bbox_inches='tight')
   
    plt.clf()
    fig.clear('all')
    plt.close(fig)
    plt.close()


# %% plotting small set of nodes in the same figure


# %%
#### load in frequency axis
# foldername = r'D:\Research\Data\Van_Allen_Probes\\'
# folder_content=os.listdir(foldername)
# data=(readsav(foldername + folder_content[2]))
# freq=np.reshape(data['freqs'][0:50],(-1,1)).T

# import matplotlib.pyplot as plt
# for i in range(0,100):    
    
#     fig, axs = plt.subplots(2, 2)
    
#     Y=np.mean(vlf_norm[0][:,np.argwhere((predictions==i))].T,axis=1)
#     err=np.std(vlf_norm[0][:,np.argwhere((predictions==i))].T,axis=1)
#     axs[0, 0].errorbar(np.squeeze((freq)), np.squeeze(Y), yerr=np.squeeze(err))
#     axs[0, 0].set_title('Avg B-spectra, node: ' + str(i) + ', # of input data: ' + str(len(np.argwhere((predictions==i) ))))
#     axs[0, 0].set_xscale('log')  
#     axs[0, 0].set_xlabel('[Hz]')
#     axs[0, 0].set_ylabel('[(nT)^2/Hz]')
    
#     Y=np.mean(vlf_norm[2][:,np.argwhere((predictions==i))].T,axis=1)
#     err=np.std(vlf_norm[2][:,np.argwhere((predictions==i))].T,axis=1)
#     axs[0, 1].errorbar(np.squeeze((freq)), np.squeeze(Y), yerr=np.squeeze(err))
#     axs[0, 1].set_title('Avg E-spectra axial')
#     axs[0, 1].set_xscale('log')  
#     axs[0, 1].set_xlabel('[Hz]')
#     axs[0, 1].set_ylabel('[(V/m)^2/Hz]')
     
#     Y=np.median(vlf[3][0:50,np.argwhere((predictions==i))].T,axis=1)
#     err=np.std(vlf[3][0:50,np.argwhere((predictions==i))].T,axis=1)
#     axs[1, 0].errorbar(np.squeeze((freq)), np.squeeze(Y), yerr=np.squeeze(err))
#     axs[1, 0].set_title('Avg compressibility')
#     axs[1, 0].set_xscale('log')  
#     axs[1, 0].set_xlabel('[Hz]')
#     axs[1, 0].set_ylabel('[ ]')
    
#     #axs[1, 1].hist(vlf[9][np.argwhere((predictions==i))] )
#     axs[1, 1].hexbin(vlf[9][np.argwhere((predictions==i))],vlf[7][np.argwhere((predictions==i))],gridsize=50, bins='log', cmap='Blues' )
#     axs[1, 1].set_title('Spatial location')
#     axs[1, 1].set_xlabel('L-shell')
#     axs[1, 1].set_ylabel('Mlt')
#     #axs[1, 1].set_xlabel('Mlt')
#     #axs[1, 1].set_ylabel('Mlat')
#     axs[1, 1].set_ylabel('# of occurrence')
    
#     fig.tight_layout()
#     plt.show()