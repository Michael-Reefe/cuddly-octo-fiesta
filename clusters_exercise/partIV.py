# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 14:49:36 2021

@author: blake
"""

import numpy as np
import pandas as pd
from scipy import optimize as opt
from matplotlib import pyplot as plt
from typing import Union, Optional, List, Tuple
def line(x, a, b):
    return a*x + b

Virgo = pd.read_csv('virgo7.5deg.opthi.csv')
notVirgo = pd.read_csv('notVirgo7.5deg.opthi.csv')

Vra = Virgo['radeg']
Vdec = Virgo['decdeg']


VL = Virgo['logL_V']
NL = notVirgo['logL_V']

Vgf = Virgo['logMH']
Ngf = notVirgo['logMH']

Vi = Virgo['logL_i']
Ni = notVirgo['logL_i']

Vgas2_i = Virgo['gas2L_i']
Ngas2_i = notVirgo['gas2L_i']




Vgas2_l = Virgo['gas2L_V'] 
Ngas2_l = notVirgo['gas2L_V'] 


fig, ax = plt.subplots(dpi = 600)
ax.scatter(VL,Vgf , c='r', s=1, label='Virgo')
ax.scatter(NL,Ngf, c='b', s=1, label='Not Virgo')
ax.set_xlabel('log V-band Luminosity')
ax.set_ylabel('Log Mass of HI')
ax.set_title('The properties of local α.40 galaxies in Virgo and outside Virgo')

popt1, pcov1 = opt.curve_fit(line, VL,Vgf )
popt2, pcov2 = opt.curve_fit(line, NL,Ngf )
fit_x = np.linspace(np.nanmin(NL), np.nanmax(NL))
fit_y1 = popt1[0]*fit_x + popt1[1]
fit_y2 = popt2[0]*fit_x + popt2[1]

ax.plot(fit_x, fit_y1, 'r--', label='Linear fits')
ax.plot(fit_x, fit_y2, 'b--')
ax.legend()



fig, ax = plt.subplots(dpi = 600)
ax.scatter(VL,Vgas2_l , c='r', s=1, label='Virgo')
ax.scatter(NL,Ngas2_l, c='b', s=1, label='Not Virgo')
ax.set_xlabel('log V-band Luminosity')
ax.set_ylabel('Log gas fraction')
ax.set_title('The properties of local α.40 galaxies in Virgo and outside Virgo')


popt1, pcov1 = opt.curve_fit(line, VL,Vgas2_l)
popt2, pcov2 = opt.curve_fit(line, NL,Ngas2_l)

fit_x = np.linspace(np.nanmin(NL), np.nanmax(NL))
fit_y1 = popt1[0]*fit_x + popt1[1]
fit_y2 = popt2[0]*fit_x + popt2[1]
ax.plot(fit_x, fit_y1, 'r--', label='Linear fits')
ax.plot(fit_x, fit_y2, 'b--')
ax.legend()


fig, ax = plt.subplots(dpi = 600)
ax.hist( [Vgas2_l,Ngas2_l] , bins=30, label=['Virgo','Not Virgo'], color=['r','b'],density = True)
ax.set_xlabel('Log of Gas fraction')
ax.set_ylabel('Probablity Density')
ax.legend()
ax.set_title('The properties of local α.40 galaxies in Virgo and outside Virgo (V-Band)')

fig, ax = plt.subplots(dpi = 600)
ax.hist( [Vgas2_i,Ngas2_i] , bins=30, label=['Virgo','Not Virgo'], color=['r','b'],density = True)
ax.set_xlabel('Log of Gas fraction')
ax.set_ylabel('Probablity Density')
ax.legend()
ax.set_title('The properties of local α.40 galaxies in Virgo and outside Virgo (I-band)')


'''
from matplotlib import pyplot as plt, contour as contour, colors as colors
fig, ax = plt.subplots(dpi = 600)
ax.scatter(Vra, Vdec, c=VL, s=1, cmap='coolwarm', norm=colors.Normalize(vmin=np.nanmin(VL), vmax=np.nanmax(VL)))
#axis.set_xlabel('$%s$ [mag]' % (xcolor[0] + ' - ' + xcolor[1]))
#axis.set_ylabel('$%s$ [mag]' % (ycolor[0] + ' - ' + ycolor[1]))


fig, ax = plt.subplots(dpi = 600)
ax.scatter(Vi,Vgf , c='r', s=1, label='Virgo')
ax.scatter(Ni,Ngf, c='b', s=1, label='Not Virgo')
ax.set_xlabel('log V-band Luminosity')
ax.set_ylabel('Log Mass of HI')
ax.set_title('The properties of local α.40 galaxies in Virgo and outside Virgo')



fig, ax = plt.subplots(dpi = 600)
ax.scatter(Vgas2_l, Vgf , c='r', s=1, label='Virgo')
ax.scatter(Ngas2_l, Ngf, c='b', s=1, label='Not Virgo')
ax.set_xlabel('gas fraction')
ax.set_ylabel('mass of hydrogen')
ax.set_title('The properties of local α.40 galaxies in Virgo and outside Virgo')



fig, ax = plt.subplots(dpi = 600)
ax.scatter(VL,Vgas2_i , c='r', s=1, label='Virgo')
ax.scatter(NL,Ngas2_i, c='b', s=1, label='Not Virgo')
ax.set_xlabel('log V-band Luminosity')
ax.set_ylabel('Log gas fraction')
ax.set_title('The properties of local α.40 galaxies in Virgo and outside Virgo')

'''