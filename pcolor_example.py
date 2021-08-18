############## initial plotting DRAFT
# inspect the data to find out a way to eliminate shadow spikes

####  plotting a few days of data
from os.path import dirname, join as pjoin
import scipy.io as sio
from scipy.io import readsav
import os

#### load in frequency axis
data=readsav(r'D:\Research\Data\Van_Allen_Probes\rbsp_EMFISIS_freq_and_bandwidth.sav')
freq=data['freqs']

#### load VLF data
foldername = r'E:\rbsp_hiss_hsr_daily_vlf_files\a\\'
folder_content=[f for f in os.listdir(foldername) if not f.startswith('.')]
folder_content.sort()

#### read in some days of data
data=[]
for i in range(1000,1010):
 data.append(readsav(foldername + folder_content[i]))

### time stamp correction
import numpy as np
import pandas as pd
NN = 719529 # NN = datenum('01-jan-1970 00:00:00','dd-mmm-yyyy HH:MM:SS')    
b_time = (data[0]['b_time']/86400) + NN
timestamps = pd.to_datetime(b_time-719529, unit='D')

#### pcolor plot
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np

# %%

i=8 # pick a day
rr=range(0,100) # pick a range within day i
var='e_spectra_axial'
plt.figure()
plt.title(var)
cs = plt.pcolormesh(timestamps[rr], np.log10(freq), np.log10( data[i][var][:,rr]),shading='flat', cmap='inferno')

p_min=5
p_max=95
plt.clim(np.percentile(np.reshape(np.log10( data[i][var]),(-1,1)),p_min), np.percentile(np.reshape(np.log10( data[i][var]),(-1,1)),p_max) )
plt.colorbar()
plt.show()


# %%
### plot single slice of a spectra

#plt.scatter(  np.log10(freq), np.log10( data[i][var][:,0]) )

X=[]
Y=[]
for j in range(0,50):
  plt.scatter(  np.log10(freq), np.log10( data[i][var][:,j]) )
  X.append(np.log10(freq))
  Y.append( np.log10( data[i][var][:,j]))
# %%

plt.hexbin(X, Y, gridsize=60, cmap='Blues')
cb = plt.colorbar(label='count in bin')

# %%
### plot a single frequency bin
plt.plot(np.log10( data[i][var][0,rr]) )
plt.plot(np.log10( data[i][var][5,rr]) )
#plt.plot(np.log10( data[i][var][10,rr]) )

# %%
plt.scatter(data[i][var][0,0:2000], data[i][var][8,0:2000] )

# %% outlier detection
## isolation forest 


