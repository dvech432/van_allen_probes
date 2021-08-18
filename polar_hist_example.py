# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 16:46:44 2021

@author: vechd
"""

# define binning
rbins = np.linspace(0, 7,10)
abins = np.linspace(-np.pi, np.pi, 30)

#calculate histogram
hist, _, _ = np.histogram2d(180*((yy-12)/12)*np.pi/180, xx, bins=(abins, rbins))
A, R = np.meshgrid(abins, rbins)

# plot
fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))

pc = ax.pcolormesh(A, R, np.log10(hist.T/np.sum(hist)), cmap="viridis",vmin=-4, vmax=-2)
#fig.colorbar(pc)

cbar = plt.colorbar(pc)
cbar.set_label('Log10 probability')

ax.grid(True)
plt.show()