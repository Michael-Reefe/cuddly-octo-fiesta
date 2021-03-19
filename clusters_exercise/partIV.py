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

Virgo = pd.read_csv('virgo7.5deg.opthi.csv')
notVirgo = pd.read_csv('notVirgo7.5deg.opthi.csv')




VL = Virgo['logL_V']
Vgf = Virgo['logMH']
NL = notVirgo['logL_V']
Ngf = notVirgo['logMH']



fig, ax = plt.subplots()
ax.scatter(VL,Vgf , c='r', s=1, label='Virgo')
ax.scatter(NL,Ngf, c='b', s=1, label='Not Virgo')
#ax.plot([11.875, 13], [16, 16], 'k--', lw=0.5, label='$\\alpha .40$ cutoff')
#ax.set_xticks(np.linspace(13, 11.875, 10))
#ax.set_xlim(13, 11.875)
ax.legend()
#ax.set_xticklabels(['$13^\\mathrm{h}$', '', '$12^\\mathrm{h}45^\\mathrm{m}$', '', '$12^\\mathrm{h}30^\\mathrm{m}$', '',
#                    '$12^\\mathrm{h}15^\\mathrm{m}$', '', '$12^\\mathrm{h}$', ''])
#yt = np.arange(6, 22, 2)
#ax.set_yticks(yt)
#ax.set_yticklabels(['$%d\\degree$' % yti for yti in yt])
ax.set_xlabel('log V-band Luminosity')
ax.set_ylabel('Log Mass of HI')
ax.set_title('The properties of local α.40 galaxies in Virgo and outside Virgo')


fig, ax = plt.subplots()

ax.hist(Vgf, bins=100, label='Virgo', color='r',density = True)
ax.hist(Ngf, bins=100, label='Not Virgo', color='b',density = True)

ax.set_xlabel('Gas fraction')
ax.set_ylabel('Probablity Density')
ax.legend()






