# loading in the density file, correct time stamp
import pandas as pd

def process_den(den_str):
  A=[]  

  NN = 719529 # NN = datenum('01-jan-1970 00:00:00','dd-mmm-yyyy HH:MM:SS')    
  #den_time = (den_str['rbsp_density_structure']['density_time'][0]/86400) + NN # old
  den_time = den_str['rbsp_density_structure']['density_time'][0] # new
  #den_str['rbsp_density_structure']['density_time'][0] = pd.to_datetime(den_time-719529, unit='D') # old
  den_str['rbsp_density_structure']['density_time'][0] = den_time # new
  A.append(den_str['rbsp_density_structure']['density_time'][0])
  A.append(den_str['rbsp_density_structure']['density_value'][0])
  return A