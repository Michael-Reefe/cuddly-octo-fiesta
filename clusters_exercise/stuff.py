import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import Union, Optional, List, Tuple

vopt = pd.read_csv('virgo7.5deg.voptlt3000.csv')
a40 = pd.read_csv('virgo7.5deg.a40lt3000.csv')

med1 = np.nanmedian(vopt['sepmin'])
med2 = np.nanmedian(a40['sepmin'])


def deg_to_hours(degrees: Union[float, int]) -> float:
    return degrees / 15


def sexagesimal(degrees: Union[float, int]) -> Tuple[int, int, float]:
    """
    :param degrees: right ascension decimal value in degrees
    :return: right ascension formatted as a sexigesimal tuple (hh, mm, ss.sss)
    """
    hours = deg_to_hours(degrees)
    hh = int(hours)
    if hh != 0:
        h_decimal = hours % hh
    else:
        h_decimal = hours
    minutes = h_decimal * 60
    mm = int(minutes)
    if mm != 0:
        m_decimal = minutes % mm
    else:
        m_decimal = minutes
    seconds = m_decimal * 60
    ss = round(seconds, 3)
    return hh, mm, ss


vopt['ra'] = pd.Series([tuple(sexagesimal(ra)) for ra in vopt['radeg']], name='ra')
a40['ra'] = pd.Series([tuple(sexagesimal(ra)) for ra in a40['radeg']], name='ra')

fig, ax = plt.subplots()
ax.scatter(vopt['radeg'] / 15, vopt['decdeg'], c='r', s=1, label='optical')
ax.scatter(a40['radeg'] / 15, a40['decdeg'], c='b', s=1, label='$\\alpha .40$')
ax.plot([11.875, 13], [16, 16], 'k--', lw=0.5, label='$\\alpha .40$ cutoff')
ax.set_xticks(np.linspace(13, 11.875, 10))
ax.set_xlim(13, 11.875)
ax.legend()
ax.set_xticklabels(['$13^\\mathrm{h}$', '', '$12^\\mathrm{h}45^\\mathrm{m}$', '', '$12^\\mathrm{h}30^\\mathrm{m}$', '',
                    '$12^\\mathrm{h}15^\\mathrm{m}$', '', '$12^\\mathrm{h}$', ''])
yt = np.arange(6, 22, 2)
ax.set_yticks(yt)
ax.set_yticklabels(['$%d\\degree$' % yti for yti in yt])
ax.set_xlabel('Right ascension ($\\alpha$, J2000)')
ax.set_ylabel('Declination ($\\delta$, J2000)')
ax.set_title('Sky Distribution of Galaxies in the Virgo Cluster')
plt.savefig('skydist.png', dpi=300)
