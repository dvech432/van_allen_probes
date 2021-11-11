## processing the lpp distance file and interpolate it to the summary file


import numpy as np
from scipy.io import readsav
data=readsav(r'D:\Research\Data\Van_Allen_Probes\RBSP_a_PP_L_various_GRADIENT_TS04D--2012-10-01--2016-08-19.sav')


q0=np.reshape(data['rbsp_pp_structure']['PP_LSTAR'][0],(-1,1))
q1=np.reshape(data['rbsp_pp_structure']['pp_lstar_time'][0],(-1,1))
Q=np.append(q1,q0,axis=1)
w0=np.reshape(data['rbsp_pp_structure']['PP_LSIMPLE'][0],(-1,1))
w1=np.reshape(data['rbsp_pp_structure']['pp_lsimple_time'][0],(-1,1))
W=np.append(w1,w0,axis=1)
e0=np.reshape(data['rbsp_pp_structure']['PP_LMCILWAIN'][0],(-1,1))
e1=np.reshape(data['rbsp_pp_structure']['PP_LMCILWAIN_TIME'][0],(-1,1))
E=np.append(e1,e0,axis=1)
#### load in Dmitri file

from scipy.io import loadmat
filename=(r'D:\Research\Data\Van_Allen_Probes\Dmitri.mat')
dd=loadmat(filename)
d_time=dd['foo'][0]


from scipy.interpolate import interp1d #interpolate vlf[7]-[9] to the time stamps at vlf[5]
f_interp = interp1d( Q[:,0],  Q[:,1],fill_value='nan',bounds_error=False,kind='nearest')
Q_interp = f_interp(d_time)

f_interp = interp1d( W[:,0],  W[:,1],fill_value='nan',bounds_error=False,kind='nearest')
W_interp = f_interp(d_time)

f_interp = interp1d( E[:,0],  E[:,1],fill_value='nan',bounds_error=False,kind='nearest')
E_interp = f_interp(d_time)

E_interp=np.interp(d_time,   E[:,0],  E[:,1])

# %%
import matplotlib.pyplot as plt
f_interp = interp1d( E[:,0],  E[:,1],kind='linear')
E2_interp = f_interp(d_time)

plt.plot(E2_interp)
plt.plot(E_interp)


np.nonzero((np.isnan(E_interp)==False) & (np.isnan(E2_interp)==False))[0]
# %% writing into file

LPP=[]
# unix time, mlt, mlat, lshell, density, node ID
LPP.append(d_time) # b time in unix format
LPP.append(Q_interp) # PP_LSTAR
LPP.append(W_interp) # PP_LSIMPLE
LPP.append(E_interp) # PP_LMCILWAIN

from scipy.io import savemat
filename=(r'D:\Research\Data\Van_Allen_Probes\LPP.mat')
savemat(filename, {"foo":LPP})

# %%
LPP=[]
# unix time, mlt, mlat, lshell, density, node ID
LPP.append(d_time) # b time in unix format
LPP.append(Q) # PP_LSTAR
LPP.append(W) # PP_LSIMPLE
LPP.append(E) # PP_LMCILWAIN

from scipy.io import savemat
filename=(r'D:\Research\Data\Van_Allen_Probes\LPP_orig.mat')
savemat(filename, {"foo":LPP})


# %% brute force interpolation


import matplotlib.pyplot as plt

### remove nan values from original data
at0=np.nonzero(np.isnan(Q[:,1])==False)[0]
Q_filtered=Q[at0,:]
at0=np.nonzero(np.isnan(W[:,1])==False)[0]
W_filtered=W[at0,:]
at0=np.nonzero(np.isnan(E[:,1])==False)[0]
E_filtered=E[at0,:]

from scipy.interpolate import interp1d #interpolate vlf[7]-[9] to the time stamps at vlf[5]
f_interp = interp1d( Q_filtered[:,0],  Q_filtered[:,1],fill_value='nan',bounds_error=False,kind='nearest')
Q_interp = f_interp(d_time)

f_interp = interp1d( W_filtered[:,0],  W_filtered[:,1],fill_value='nan',bounds_error=False,kind='nearest')
W_interp = f_interp(d_time)

f_interp = interp1d( E_filtered[:,0],  E_filtered[:,1],fill_value='nan',bounds_error=False,kind='nearest')
E_interp = f_interp(d_time)

LPP=[]
# unix time, mlt, mlat, lshell, density, node ID
LPP.append(d_time) # b time in unix format
LPP.append(Q_interp) # PP_LSTAR
LPP.append(W_interp) # PP_LSIMPLE
LPP.append(E_interp) # PP_LMCILWAIN

from scipy.io import savemat
filename=(r'D:\Research\Data\Van_Allen_Probes\LPP_v2.mat')
savemat(filename, {"foo":LPP})