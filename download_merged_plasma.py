##### TRASH

# day of

import requests
from datetime import timedelta
from datetime import date
from scipy.io import savemat

delta1= timedelta(days=1)
start_date=date.fromisoformat('2013-01-01') #start date of the merged plasma data
current_date=start_date
end_date=date.fromisoformat('2018-12-31') 


for i in range(0,(end_date-start_date).days):
 a = str(current_date.year) + str(current_date.month).zfill(2) + str(current_date.day).zfill(2)
 base_url='https://rbsp-ect.lanl.gov/data_pub/rbspa/ECT/level2/2013/rbspa_ect-elec-L2_' + a + '_v1.0.0.cdf'
 r = requests.get(base_url)
 file_name=r'D:\Research\Data\Van_Allen_Probes\Merged_plasma\\' + a + '.mat'
 savemat(file_name, r)
 current_date=current_date+delta1
 
 
 ## load in mat files
 
from os.path import dirname, join as pjoin
import scipy.io as sio
import numpy as np
mat_contents = sio.loadmat(r'D:\Research\Data\Van_Allen_Probes\Merged_plasma_mat\20130101.mat')

sorted(mat_contents.keys())

np.shape(mat_contents['FESA'][0][17][0])


matlab_datenum=mat_contents['FESA'][0][17][200]
import datetime 
python_datetime = datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum%1) - timedelta(days = 366)


import numpy as np
import pandas as pd
datenums = np.squeeze(np.array(mat_contents['FESA'][0][17]),axis=1)
timestamps = pd.to_datetime(datenums-719529, unit='D')

index = pd.DatetimeIndex(timestamps)
index.astype(np.int64)


## go from datetime to unix time
# from datetime import datetime
# dates = [datetime(2012, 5, 1), datetime(2012, 5, 2), datetime(2012, 5, 3)]
# index = pd.DatetimeIndex(dates)
# index.astype(np.int64)