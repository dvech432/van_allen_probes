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
vlf_norm=[]
vlf_norm.append(np.log10(vlf[0][0:50,:]))
vlf_norm.append(np.log10(vlf[1][0:50,:]))
vlf_norm.append(np.log10(vlf[2][0:50,:]))
## train model

from minisom import MiniSom    
som = MiniSom(10, 10, np.shape(vlf_norm)[1], sigma=1, learning_rate=0.1) # 
som.train(vlf_norm[2].T, 5000) # 

## plot distribution of input vectors among the nodes
import matplotlib.pyplot as plt
w_x, w_y = zip(*[som.winner(d) for d in vlf_norm[2].T])
w_x = np.array(w_x)
w_y = np.array(w_y)

plt.figure(figsize=(10, 9))
plt.pcolor(som.distance_map().T, cmap='bone_r', alpha=.2)
plt.colorbar()

plt.scatter(w_x+.5+(np.random.rand(len(w_x))-.5)*.8,
                w_y+.5+(np.random.rand(len(w_x))-.5)*.8, 
                s=50)
plt.legend(loc='upper right')
plt.grid()
#plt.savefig('resulting_images/som_seed.png')
plt.show()

# plot average power spectra in each node

#### load in frequency axis
foldername = r'D:\Research\Data\Van_Allen_Probes\\'
folder_content=os.listdir(foldername)
data=(readsav(foldername + folder_content[2]))
freq=np.reshape(data['freqs'][0:50],(-1,1)).T

for i in range(0,10):
  for j in range(0,10):          
      if len(np.argwhere((w_x==i) & (w_y==j)))>5000:
        Y=np.mean(vlf_norm[2][:,np.argwhere((w_x==i) & (w_y==j))].T,axis=1)
        err=np.std(vlf_norm[2][:,np.argwhere((w_x==i) & (w_y==j))].T,axis=1)
      
        plt.errorbar(np.squeeze(np.log10(freq)), np.squeeze(Y), yerr=np.squeeze(err)) 
        plt.title('Node id: (' + str(i) + ',' +str(j) + '), # of input data: ' + str(len(np.argwhere((w_x==i) & (w_y==j)))) )
        #plt.show()
      #plt.savefig('resulting_images/som_seed.png')
      
# %% visual inspection of the nodes
#### load in frequency axis
foldername = r'D:\Research\Data\Van_Allen_Probes\\'
folder_content=os.listdir(foldername)
data=(readsav(foldername + folder_content[2]))
freq=data['freqs']


for i in range(0,100):
  if w_x[i]==9:  
    plt.scatter(  np.log10(freq), np.log10( vlf[2][:,i]), c='blue' )
  if w_x[i]==8:
    plt.scatter(  np.log10(freq), np.log10( vlf[2][:,i]), c='red' )

