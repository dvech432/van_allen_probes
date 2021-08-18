# 1) take the read-in data from som_data_prep
# 2) load in the trained shadow spike removal SOM
# 3) delete bad data and return cleaned data set
def shadow_removal(vlf, den_interp, kp_interp, dst_interp):

  import numpy as np

  # prepare data for the SOM
  vlf_norm=[]
  vlf_norm.append(np.log10(vlf[0][0:24,:])) #vlf0['b_spectra']
  vlf_norm.append(np.log10(vlf[1][0:24,:])) #vlf0['e_spectra_spin']
  vlf_norm.append(np.log10(vlf[2][0:24,:])) #vlf0['e_spectra_axial']    
  # load in the pre-trained model  
  import pickle
  filename=(r'D:\Research\Data\Van_Allen_Probes\Models\model_shadow.sav')   
  vlf_som = pickle.load(open(filename, 'rb'))
  predictions = vlf_som.predict(vlf_norm[2].T) #[2] <-- use axial E-field

  shadow_node=np.array([0,1,2,10,11,12,20,21,22,30,31,40,50]) # list of node id with shadow spike
  # generate array with indices that will be deleted
  del_shadow=[]
  for i in range(0,len(predictions)):
    if np.sum(predictions[i]==shadow_node)>0: # if a given node number is element of shadow_node
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
  
  
  return vlf, den_interp, kp_interp, dst_interp