
#### checking files, initial inspection

from os.path import dirname, join as pjoin
import scipy.io as sio
from scipy.io import readsav
import os

### checking primary folder
foldername = r'E:\rbsp_hiss_hsr_daily_vlf_files\a\\'
folder_content=os.listdir(foldername)

#### read in some days of data
data=[]
for i in range(0,10):
 data.append(readsav(foldername + folder_content[i]))

import numpy as np
print(data[0].keys())

### plotting some power spectra B and E-fields
import matplotlib.pyplot as plt
plt.plot(data[0]['t_correct'][2])


## checking secondary folder
### checking primary folder
foldername = r'E:\data_TS04D_all_L\a\\'
folder_content=os.listdir(foldername)

#### read in some days of data
data=[]
for i in range(0,10):
 if folder_content[i][-3:]=='sav': # read in only sav files, ignore png extension
   data.append(readsav(foldername + folder_content[i]))
   
   
print(data[0].keys())

   