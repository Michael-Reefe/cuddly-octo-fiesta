import numpy as np
import scipy.stats as st
import scipy.optimize as opt
from matplotlib import pyplot as plt
from matplotlib import markers as mpl_markers
from matplotlib import ticker as ticker
import pandas as pd
import re
import string
import os
import requests
import itertools

galaxy_data = pd.read_csv('notebook.csv', delimiter=',', header=0)
galaxy_data['u_g'] = galaxy_data['u'] - galaxy_data['g']
galaxy_data['g_r'] = galaxy_data['g'] - galaxy_data['r']
for i in range(len(galaxy_data)):
    if galaxy_data.loc[i, 'redshift'] != 'Null':
        galaxy_data.loc[i, 'redshift'] = float(galaxy_data.loc[i, 'redshift'])
    else:
        galaxy_data.loc[i, 'redshift'] = np.nan

# Morphology type sorting
morphologies = pd.read_csv('morphologies.csv', delimiter=',', header=0)
sizes = pd.Series([], name='size', dtype=int)
for i, morphology in enumerate(morphologies.iloc(0)):
    number = re.search('([1-3])', morphology[0])
    size = np.nan
    if number:
        size = number.group(0)
        morphology[0] = morphology[0].replace(str(size), '')
    if np.in1d(morphology[0], ['({})'.format(l) for l in string.ascii_lowercase]):
        morphology[0] = 'Unc'
    sizes[i] = size

frames = [galaxy_data, morphologies, sizes]
galaxy_data_1 = pd.concat(frames, axis=1)
morph_types = np.unique(galaxy_data_1['morphology'])

valid_markers = ['x', '1', '+', '2', '^', '3', 'v', '4', 'P', 'X', 'o', 'd', '.', '<', '>']
markers = valid_markers[0:len(morph_types)]


def plot_formatter(ax, title='Title', fname='fname.png'):
    ax.set_title(title)
    ax.set_xlabel('$u - g$ [mag]')
    ax.set_ylabel('$g - r$ [mag]')
    ax.legend()
    ax2 = ax.twinx()
    ax2.set_ylabel('Blue $\\longrightarrow$ Red')
    ax2.plot([1], [1], markersize=0)
    ax3 = ax.twiny()
    ax3.set_xlabel('Blue $\\longrightarrow$ Red')
    ax3.plot([1], [1], markersize=0)
    ax2.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax3.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax2.yaxis.set_major_formatter(ticker.NullFormatter())
    ax3.xaxis.set_major_formatter(ticker.NullFormatter())
    plt.savefig(fname, dpi=300)
    plt.close()


# For color-color diagram on each classification
plt.style.use('seaborn-dark')
fig, ax = plt.subplots()
for mtype, marker in zip(morph_types, markers):
    colorstr = 'r' if 'R' in mtype else 'b' if 'B' in mtype else 'k'
    ax.scatter(galaxy_data_1['u_g'], galaxy_data_1['g_r'], marker=marker, c=colorstr,
               s=list(map(lambda i, h: np.nanmax([40*np.float(h), 40]) if i == mtype else 0, galaxy_data_1['morphology'], galaxy_data_1['size'])),
               label=mtype.replace('a', '$\\alpha$'))
plot_formatter(ax=ax, title='Color-Color Diagram for Sample Galaxies', fname='color_color_diagram.png')


# With other surveys
alfalfa_data = pd.read_csv('alfalfa_1228+15_notebook.csv', delimiter=',', header=0)
alfalfa_data['u_g'] = alfalfa_data['u'] - alfalfa_data['g']
alfalfa_data['g_r'] = alfalfa_data['g'] - alfalfa_data['r']
# for i in range(len(alfalfa_data)):
#     if alfalfa_data.loc[i, 'redshift'] != 'Null':
#         alfalfa_data.loc[i, 'redshift'] = float(alfalfa_data.loc[i, 'redshift'])
sdss_data = pd.read_csv('SDSS_1228+15_notebook.csv', delimiter=',', header=0)
sdss_data['u_g'] = sdss_data['u'] - sdss_data['g']
sdss_data['g_r'] = sdss_data['g'] - sdss_data['r']
# for i in range(len(sdss_data)):
#     if sdss_data.loc[i, 'redshift'] != 'Null':
#         sdss_data.loc[i, 'redshift'] = float(sdss_data.loc[i, 'redshift'])

fig, ax = plt.subplots()
colors = ['b', 'r', 'k']
labels = ['Our galaxies', 'Alfalfa', 'SDSS']
for i, data in enumerate((galaxy_data, alfalfa_data, sdss_data)):
    ax.scatter(data['u_g'], data['g_r'], marker=valid_markers[i], c=colors[i], label=labels[i], s=20)
plot_formatter(ax=ax, title='Color-Color Diagram with Alfalfa and SDSS Galaxies', fname='color_color_diagram_alfalfa_sdss.png')

# Statistical data
median_gal = [np.nanmedian(galaxy_data['u_g']), np.nanmedian(galaxy_data['g_r'])]
median_alf = [np.nanmedian(alfalfa_data['u_g']), np.nanmedian(alfalfa_data['g_r'])]
median_sdss = [np.nanmedian(sdss_data['u_g']), np.nanmedian(sdss_data['g_r'])]
medians = np.vstack((median_gal, median_alf, median_sdss))

np.savetxt('median_values.txt', medians, delimiter=',', header='med(u-g),med(g-r)', fmt='%.5f')


# Galaxies overlap in Alfalfa/SDSS
good_alf = np.where(np.in1d(alfalfa_data['ra'], sdss_data['ra']) & np.in1d(alfalfa_data['dec'], sdss_data['dec']))[0]
good_sdss = np.where(np.in1d(sdss_data['ra'], alfalfa_data['ra']) & np.in1d(sdss_data['dec'], sdss_data['dec']))[0]

alf_overlap = alfalfa_data.iloc(0)[good_alf]
sdss_overlap = sdss_data.iloc(0)[good_sdss]

fig, ax = plt.subplots()
colors = ['b', 'r']
labels = ['Alfalfa Color', 'SDSS Color']
for i, data in enumerate((alf_overlap, sdss_overlap)):
    ax.scatter(data['u_g'], data['g_r'], marker=valid_markers[i], color='b' if i == 0 else 'r',
               label='Alfalfa' if i == 0 else 'SDSS')
plot_formatter(ax=ax, title='Color-Color Diagram with Overlapping Alfalfa/SDSS Galaxies', fname='color_color_diagram_overlap.png')


# Absolute M_r vs. u - r color
hubble_data = pd.read_csv('galaxy_search2.csv', delimiter=',', header=0)

# In km/s
c = 2.99792458e10 / (100 * 1000)
# In km/s/Mpc
H0 = 74
for data in (galaxy_data, hubble_data):
    data['distance'] = c * data['redshift'] / H0
    data['absolute_mag_r'] = data['r'] - 5*np.log10(np.array(data['distance'] * 1e6, dtype=float)) + 5
    data['u_r'] = data['u'] - data['r']

x = np.array(hubble_data['absolute_mag_r'], dtype=float)
y = np.array(hubble_data['u_r'], dtype=float)

good = np.where(np.isfinite(x) & np.isfinite(y))[0]
x = x[good]
y = y[good]

deltaX = (max(x) - min(x))/10
deltaY = (max(y) - min(y))/10
xmin = min(x) - deltaX
xmax = max(x) + deltaX
ymin = min(y) - deltaY
ymax = max(y) + deltaY

# Create meshgrid
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = st.gaussian_kde(values)
f = np.reshape(kernel(positions).T, xx.shape)

plt.style.use('default')
fig, ax = plt.subplots()
ax.set_xlim(-24, -17)
ax.set_ylim(0.5, 4)
cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
cset = ax.contour(xx, yy, f, colors='k')
# ax.clabel(cset, inline=1, fontsize=10)
pos1 = ax.get_position()
cbar_ax = fig.add_axes([0.85, pos1.y0, 0.025, pos1.height])
cbar = fig.colorbar(cfset, cax=cbar_ax, label='Probability Density')
ax.scatter(galaxy_data['absolute_mag_r'], galaxy_data['u_r'], marker=valid_markers[0], c='#00fff7', label='Our galaxies', s=40)
ax.legend()
ax.set_xlabel('$M_r$ [mag]')
ax.set_ylabel('$u - r$ [mag]')
ax.set_title('2D Gaussian Kernel density estimation of galaxy populations')
fig.subplots_adjust(right=0.8)
plt.savefig('color_magnitude_density.png', dpi=300, bbox_extra_artist=(cbar,))
plt.close()

fig, ax = plt.subplots()
colors = ['r', 'b']
labels = ['Hubble', 'Our galaxies']
for i, data in enumerate((hubble_data, galaxy_data)):
    ax.scatter(data['absolute_mag_r'], data['u_r'], marker='.', c=colors[i], label=labels[i], s=.5)
ax.set_title('Color-Magnitude Diagram for Galaxies')
ax.set_xlabel('$M_{r}$')
ax.set_ylabel('$u - r$ [mag]')
ax.legend()
plt.savefig('color_magnitude_diagram.png', dpi=300)
plt.close()


# Hubble distance stuff
def distance(r, C):
    m = C - 5*np.log10(r)
    return 10**(4.04 + 0.2*m)


fit = opt.curve_fit(distance, hubble_data['petroR50_r'], hubble_data['distance'] * 1e6)
C1 = fit[0][0]
fit2 = opt.curve_fit(distance, hubble_data['petroR90_r'], hubble_data['distance'] * 1e6)
C2 = fit2[0][0]
r_range = np.geomspace(np.nanmin(hubble_data['petroR50_r']), np.nanmax(hubble_data['petroR50_r']), 10000)
r_range2 = np.geomspace(np.nanmin(hubble_data['petroR90_r']), np.nanmax(hubble_data['petroR90_r']), 10000)
D1_model = distance(r_range, C1)
D2_model = distance(r_range2, C2)

rms1 = np.sqrt(np.mean((hubble_data['distance'] * 1e6 - D1_model)**2)) / 1e6
rms2 = np.sqrt(np.mean((hubble_data['distance'] * 1e6 - D2_model)**2)) / 1e6

fig, ax = plt.subplots()
ax.loglog(hubble_data['petroR50_r'], hubble_data['distance'] * 1e6, 'r.', label='R50 Redshifts')
ax.loglog(r_range, D1_model, 'b-', label='Size relationship for petroR50: C = {:.5f}'.format(C1))
ax.set_title('Hubble distances, RMS = {:.5f} Mpc'.format(rms1))
ax.set_ylabel('$\\mathrm{Distance} \\mathrm{[pc]}$')
ax.set_xlabel('Radius [arcseconds]')
ax.legend()
plt.savefig('hubble_dsitances_R50.png', dpi=300)
plt.close()
cov1 = np.cov(hubble_data['distance'], hubble_data['petroR50_r'])

fig, ax = plt.subplots()
ax.loglog(hubble_data['petroR90_r'], hubble_data['distance'] * 1e6, 'y.', label='R90 Redshifts')
ax.loglog(r_range2, D2_model, 'g-', label='Size relationship for petroR90: C = {:.5f}'.format(C2))
ax.set_title('Hubble distances, RMS = {:.5f} Mpc'.format(rms2))
ax.set_ylabel('$\\mathrm{Distance} \\mathrm{[pc]}$')
ax.set_xlabel('Radius [arcseconds]')
ax.legend()
plt.savefig('hubble_dsitances_R90.png', dpi=300)
plt.close()
cov2 = np.cov(hubble_data['distance'], hubble_data['petroR90_r'])

print('Covariance for {} and {}: {}'.format('Distance', 'petroR50_r', cov1))
print('Covariance for {} and {}: {}'.format('Distance', 'petroR90_r', cov2))

# With extra fit parameters
def mag(r, C):
    return C - 5*np.log10(r)

fitmag = opt.curve_fit(mag, hubble_data['petroR50_r'], hubble_data['r'])
def distance2(r, C_2):
    m = mag(r, fitmag[0][0])
    return 10 ** (C_2 + 0.2*m)

for band in [hubble_data[b] for b in ['r', 'u', 'g', 'i', 'z']]:
    fig, ax = plt.subplots()
    band_label = 'r [mag]' if np.array_equal(band, hubble_data['r']) else 'u [mag]' if np.array_equal(band, hubble_data['u']) else 'g [mag]' if np.array_equal(band, hubble_data['g']) \
                  else 'i [mag]' if np.array_equal(band, hubble_data['i']) else 'z [mag]'
    for i, radius in enumerate([hubble_data['petroR50_r'], hubble_data['petroR90_r']]):
        Ci, _ = opt.curve_fit(mag, radius, band)
        ri = np.geomspace(np.nanmin(radius), np.nanmax(radius), 10000)
        model = mag(ri, Ci[0])
        fmtstr = 'r.' if i == 0 else 'y.'
        ax.semilogx(radius, band, fmtstr, label='Redshifts R50' if i == 0 else 'Redshifts R90')
        fmtstr = 'b-' if i == 0 else 'g-'
        mod = 'Model R50' if i == 0 else 'Model R90'
        mod += ' C = {:.5f}'.format(Ci[0])
        ax.semilogx(ri, model, fmtstr, label=mod)
        cov = np.cov(band, radius)
        print('Covariance for {} and {}: {}'.format(band_label, mod, cov))
    ax.set_title('Size-Magnitude Relationship')
    ax.set_xlabel('Radius [arcseconds]')
    ax.set_ylabel(band_label)
    ax.legend()
    plt.savefig('size_magnitude_plot_{}.png'.format(band_label.replace(' [mag]', '')), dpi=300)
    plt.close()

def distance2(m, C_2):
    return 10 ** (C_2 + 0.2*m)

fit = opt.curve_fit(distance2, hubble_data['r'], hubble_data['distance'] * 1e6)

fit2 = opt.curve_fit(distance2, hubble_data['u'], hubble_data['distance'] * 1e6)

r_range = np.geomspace(np.nanmin(hubble_data['r']), np.nanmax(hubble_data['r']), 10000)
r_range2 = np.geomspace(np.nanmin(hubble_data['u']), np.nanmax(hubble_data['u']), 10000)
D1_model_2 = distance2(r_range, *fit[0])
D2_model_2 = distance2(r_range2, *fit2[0])
rms1_2 = np.sqrt(np.mean((hubble_data['distance'] * 1e6 - D1_model_2)**2)) / 1e6
rms2_2 = np.sqrt(np.mean((hubble_data['distance'] * 1e6 - D2_model_2)**2)) / 1e6

fig, ax = plt.subplots()
ax.semilogy(hubble_data['r'], hubble_data['distance'] * 1e6, 'r.', label='r magnitudes')
ax.semilogy(r_range, D1_model_2, 'b-', label='Distance relationship for r mag')
ax.set_title('Hubble distances, RMS = {} Mpc'.format(rms1_2))
ax.set_ylabel('$\\mathrm{Distance} \\mathrm{[pc]}$')
ax.set_xlabel('r [mag]')
ax.legend()
ax.invert_xaxis()
plt.savefig('hubble_dsitances_R50_morefit.png', dpi=300)
plt.close()
cov1_2 = np.cov(hubble_data['distance'], hubble_data['r'])

fig, ax = plt.subplots()
ax.semilogy(hubble_data['u'], hubble_data['distance'] * 1e6, 'y.', label='u magnitudes')
ax.semilogy(r_range2, D2_model_2, 'g-', label='Distance relationship for u mag')
ax.set_title('Hubble distances, RMS = {} Mpc'.format(rms2_2))
ax.set_ylabel('$\\mathrm{Distance} \\mathrm{[pc]}$')
ax.set_xlabel('u [mag]')
ax.legend()
ax.invert_xaxis()
plt.savefig('hubble_dsitances_R90_morefit.png', dpi=300)
plt.close()
cov2_2 = np.cov(hubble_data['distance'], hubble_data['u'])

print('Covariance for {} and {}: {}'.format('Distance', 'r [mag]', cov1_2))
print('Covariance for {} and {}: {}'.format('Distance', 'u [mag]', cov2_2))

np.savetxt('fit_parameters.txt', np.array([[fitmag[0][0], fit[0][0]], [fitmag[0][0], fit2[0][0]]]), delimiter=',', header='C, C_2', fmt=('%.5f'))

# METHOD 1: JUST FIND CLOSEST GALAXIES TO FIT
diffs = np.array(np.abs(hubble_data['distance']*1e6 - distance(hubble_data['petroR90_r'], C2)))
a = np.argsort(diffs)
min_galaxies = hubble_data.loc[a[0:100]]

fig, ax = plt.subplots()
ax.loglog(min_galaxies['petroR90_r'], min_galaxies['distance']*1e6, '.')
ax.set_ylim(1e4, 1e10)
plt.savefig('min_galaxies.png', dpi=300)
plt.close()
min_galaxies.to_csv('min_galaxies.csv', index=False)

fig, ax = plt.subplots()
ax.scatter(min_galaxies['u'] - min_galaxies['g'], min_galaxies['g'] - min_galaxies['r'], c='r', marker='x')
plot_formatter(ax, title='Closest galaxies to distance fit relationship', fname='min_galaxies_color.png')

min_galaxies['absolute_mag_r'] = min_galaxies['r'] - 5*np.log10(np.array(min_galaxies['distance'] * 1e6, dtype=float)) + 5

plt.style.use('default')
fig, ax = plt.subplots()
ax.set_xlim(-24, -17)
ax.set_ylim(0.5, 4)
cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
cset = ax.contour(xx, yy, f, colors='k')
# ax.clabel(cset, inline=1, fontsize=10)
pos1 = ax.get_position()
cbar_ax = fig.add_axes([0.85, pos1.y0, 0.025, pos1.height])
cbar = fig.colorbar(cfset, cax=cbar_ax, label='Probability Density')
ax.scatter(min_galaxies['absolute_mag_r'], min_galaxies['u'] - min_galaxies['r'], marker=valid_markers[0], c='#00fff7', label='Closest galaxies', s=40)
ax.legend()
ax.set_xlabel('$M_r$ [mag]')
ax.set_ylabel('$u - r$ [mag]')
ax.set_title('Closest galaxies to distance fit relation in color-magnitude')
fig.subplots_adjust(right=0.8)
plt.savefig('min_galaxies_colormag_density.png', dpi=300, bbox_extra_artist=(cbar,))
plt.close()
# GALAXIES HAVE LITTLE IN COMMON

# METHOD 2: PICK ONLY GALAXIES CLOSE TO DENSITY PEAKS
good = np.where((hubble_data['u_r'] <= 2.9) & (hubble_data['u_r'] >= 2.7) & (hubble_data['absolute_mag_r'] >= -21.5) & (hubble_data['absolute_mag_r'] <= -21)
                & (hubble_data['r'] < 17))
ellipticals = hubble_data.loc[good]
ellipticals = ellipticals.reset_index(drop=True)
good = np.where((hubble_data['u_r'] <= 1.9) & (hubble_data['u_r'] >= 1.7) & (hubble_data['absolute_mag_r'] >= -20.9) & (hubble_data['absolute_mag_r'] <= -20.3)
                & (hubble_data['r'] < 17))
spirals = hubble_data.loc[good]
spirals = spirals.reset_index(drop=True)

# diffs2 = np.array(np.abs(ellipticals['distance']*1e6 - distance(ellipticals['petroR90_r'], C2)))
# a = np.argsort(diffs2)
# ellipticals = ellipticals.loc[a[0:100]]

# diffs3 = np.array(np.abs(spirals['distance']*1e6 - distance(spirals['petroR90_r'], C2)))
# a = np.argsort(diffs3)
# spirals = spirals.loc[a[0:100]]

fig, ax = plt.subplots()
rmsloss_ellip = np.sqrt(np.mean((ellipticals['distance'] * 1e6 - distance(ellipticals['petroR90_r'], C2))**2)) / 1e6
rmsloss_spir = np.sqrt(np.mean((spirals['distance'] * 1e6 - distance(spirals['petroR90_r'], C2))**2)) / 1e6
ax.loglog(ellipticals['petroR90_r'], ellipticals['distance'] * 1e6, 'r.', label='Early type R90 Redshifts.  RMS = {:.3f} Mpc'.format(rmsloss_ellip))
ax.loglog(spirals['petroR90_r'], spirals['distance'] * 1e6, 'b.', label='Late type R90 Redshifts.  RMS = {:.3f} Mpc'.format(rmsloss_spir))
ax.loglog(r_range2, D2_model, 'k-', label='Size relationship for petroR90')
ax.set_title('Hubble distances')
ax.set_ylabel('$\\mathrm{Distance} \\mathrm{[pc]}$')
ax.set_xlabel('Radius [arcseconds]')
ax.legend()
plt.savefig('ellipticals_spirals_distance_fit.png', dpi=300)
plt.close()

e_frames = [ellipticals.loc[:, 'objid'], ellipticals.loc[:, 'ra'], ellipticals.loc[:, 'dec']]
e = pd.concat(e_frames, axis=1)
s_frames = [spirals.loc[:, 'objid'], spirals.loc[:, 'ra'], spirals.loc[:, 'dec']]
s = pd.concat(s_frames, axis=1)
e.to_csv('early_types_cut.csv', index=False)
s.to_csv('late_types_cut.csv', index=False)
cov_ellip = np.cov(ellipticals['distance'], ellipticals['petroR90_r'])
cov_spir = np.cov(spirals['distance'], spirals['petroR90_r'])

print('Covariance for early-type galaxies (distance and petroR90_r): {}'.format(cov_ellip))
print('Covariance for late-type galaxies (distance and petroR90_r): {}'.format(cov_spir))

# METHOD 3: PICK GALAXIES WITH SIMILAR INTRINSIC SIZES (this relies on the redshift as a measure of distance)
hubble_data['size'] = np.radians(hubble_data['petroR90_r']/3600) * hubble_data['distance'] * 1e6

#9100
rmslosses = np.ndarray(shape=(len(hubble_data),))
for i in range(len(hubble_data)):
    good = np.where(np.isclose(hubble_data.loc[i, 'size'], hubble_data['size'], 1e-03))[0]
    size_gals = hubble_data.loc[good]
    rmslosses[i] = np.sqrt(np.mean((size_gals['distance'] * 1e6 - distance(size_gals['petroR90_r'], C2))**2)) / 1e6
min_ind = np.nanargmin(rmslosses)
good_final = np.where(np.isclose(hubble_data.loc[min_ind, 'size'], hubble_data['size'], 1e-03))[0]
size_gals = hubble_data.loc[good_final]
rmsloss_size = rmslosses[min_ind]

fig, ax = plt.subplots()
ax.loglog(size_gals['petroR90_r'], size_gals['distance'] * 1e6, 'r.', label='Galaxies with similar sizes, R90 Redshifts.  RMS = {:.3f} Mpc'.format(rmsloss_size))
ax.loglog(r_range2, D2_model, 'k-', label='Size relationship for petroR90')
ax.set_title('Hubble distances')
ax.set_ylabel('$\\mathrm{Distance} \\mathrm{[pc]}$')
ax.set_xlabel('Radius [arcseconds]')
ax.legend()
plt.savefig('size_gals_distance_fit.png', dpi=300)
plt.close()
size_gals.to_csv('size_gals.csv', index=False)
ss_frames = [size_gals.loc[:, 'objid'], size_gals.loc[:, 'ra'], size_gals.loc[:, 'dec']]
sss = pd.concat(ss_frames, axis=1)
sss.to_csv('size_gals_cut.csv', index=False)
cov_size = np.cov(size_gals['distance'], size_gals['petroR90_r'])
print('Covariance for size-related galaxies (distance and petroR90_r): {}'.format(cov_size))


# "absolute angular radius" for 1000 Mpc
def distance3(r, C, B):
    m = C - B*np.log10(r)
    return 10**(4.04 - (1/B)*m)

hubble_data['absolute_angular_size'] = np.degrees(hubble_data['size'] / (1000 * 1e6)) * 3600
fit = opt.curve_fit(distance3, hubble_data['absolute_angular_size'], hubble_data['distance'] * 1e6)
r_range3 = np.geomspace(np.nanmin(hubble_data['absolute_angular_size']), np.nanmax(hubble_data['absolute_angular_size']), 10000)
D3_model = distance3(r_range3, *fit[0])

fig, ax = plt.subplots()
ax.loglog(hubble_data['absolute_angular_size'], hubble_data['distance'] * 1e6, 'r.', label='Redshift Distances')
# ax.loglog(r_range3, D3_model, 'b-', label='Size relationship for absolute angular size')
ax.set_title('Hubble distances')
ax.set_ylabel('$\\mathrm{Distance}\\ \\mathrm{[pc]}$')
ax.set_xlabel('Absolute Angular Radius [arcseconds]')
ax.legend()
plt.savefig('hubble_dsitances_absolute.png', dpi=300)
plt.close()

# import webbrowser
# # for i, galaxy in enumerate(pd.concat([alfalfa_data, sdss_data], ignore_index=True, sort=False).iloc(0)):
# for i, galaxy in enumerate(sdss_data.iloc(0)):
#     s = requests.Session()
#     url = 'http://skyserver.sdss.org/dr16/en/tools/chart/navi.aspx?'\
#               'ra={}&dec={}&scale=0.2&width=120&height=120&opt='.format(galaxy['ra'], galaxy['dec'])
#     webbrowser.open(url, new=2, autoraise=True)
    # if not os.path.exists('Galaxy_{}'.format(i + 1)):
    #     os.mkdir('Galaxy_{}'.format(i + 1))
    # urldata = s.get(url='http://skyserver.sdss.org/dr16/en/tools/chart/navi.aspx?'
    #           'ra={}&dec={}&scale=0.2&width=120&height=120&opt='.format(galaxy['ra'], galaxy['dec']),
    #       headers={'User-Agent': 'Mozilla/5.0'})
#     expl_id = re.search('javascript:explore\(\'([0-9]+)\'\)', urldata.text)
#     if expl_id:
#         expl_id = expl_id.group(1)
#         infodata = s.get('http://skyserver.sdss.org/dr16/en/tools/explore/Summary.aspx?id={}'.format(expl_id),
#                          headers={'User-Agent': 'Mozilla/5.0'})
#         img_urls = re.search('<img alt src=\"(.+?)\"', infodata.text)
#         if img_urls:
#             galaxy_pic = img_urls.group(1)
#             spectrum_pic = img_urls.group(2)
#             image = s.get(galaxy_pic, headers={'User-Agent': 'Mozilla/5.0'})
#             with open('Galaxy_{}'.format(i + 1) + os.sep + expl_id + '.png', 'wb') as file:
#                 file.write(image.content)
#             spectrum_pic = 'http://skyserver.sdss.org/dr16/en/' + spectrum_pic.replace('../../', '')
#             image2 = s.get(spectrum_pic, headers={'User-Agent': 'Mozilla/5.0'})
#             with open('Galaxy_{}'.format(i + 1) + os.sep + expl_id + '_spectrum.png', 'wb') as file:
#                 file.write(image2.content)
#     breakpoint()