import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt, contour as contour, colors as colors
import itertools
from typing import Union, Optional, Tuple, List

data = pd.read_csv('query_result.csv', delimiter=',', header=0)
data['ug'] = data['modelmag_u'] - data['modelmag_g']
data['gr'] = data['modelmag_g'] - data['modelmag_r']
data['ri'] = data['modelmag_r'] - data['modelmag_i']
data['iz'] = data['modelmag_i'] - data['modelmag_z']

# Example plot for exercise 3
plt.style.use('ggplot')
fig, ax = plt.subplots()
data.plot(x='ug', y='gr', kind='scatter', xlabel='$u - g$ [mag]', ylabel='$g - r$ [mag]', title='Quasar Color-Color Diagram')
plt.savefig('quasar-color-color.png', dpi=300)
plt.close()

# Plots for exercise 4
# Scatterplots
x_data = np.array(['ug', 'gr', 'ri', 'iz'], dtype=str).reshape((2, 2))
fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 8), sharey=True)
for i, j in itertools.product(range(2), range(2)):
    data.plot(x=x_data[i, j], y='z', kind='scatter', ax=axes[i, j], xlabel='$%s$ [mag]' % (x_data[i, j][0] + ' - ' + x_data[i, j][1]),
              ylabel='Redshift')
fig.suptitle('Quasar Color-Redshift Diagrams')
plt.savefig('quasar-color-redshift.png', dpi=300)
plt.close()


# Density plots - LINEAR SPACING then LOGARITHMIC SPACING
def make_density_plot(norm, label: str = None, **kwargs):
    assert type(norm) in (colors.Normalize, colors.LogNorm, colors.SymLogNorm, colors.NoNorm, colors.DivergingNorm,
                          colors.BoundaryNorm, colors.PowerNorm, colors.TwoSlopeNorm), "norm must be a matplotlib norm object"
    assert type(label) == str, "label must be a string"
    plt.style.use('default')
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 8), sharey=True)
    cfset = np.ndarray(shape=(2, 2), dtype=contour.QuadContourSet)
    for i, j in itertools.product(range(2), range(2)):
        x = np.array(data[x_data[i, j]], dtype=np.float64)
        y = np.array(data['z'], dtype=np.float64)

        good = np.where(np.isfinite(x) & np.isfinite(y))[0]
        x = x[good]
        y = y[good]

        deltaX = (np.nanmax(x) - np.nanmin(x)) / 10
        deltaY = (np.nanmax(y) - np.nanmin(y)) / 10
        ymin = 0
        ymax = np.nanmax(y) + deltaY
        xmin = np.nanmin(x) - deltaX
        xmax = np.nanmax(x) + deltaX
        if 'xlim' in kwargs.keys():
            assert len(kwargs['xlim']) == 2, "x should be of the form (xmin, xmax)"
            xlim = kwargs['xlim']
            xmin = xlim[0]
            xmax = xlim[1]
            axes[i, j].set_xlim(xmin, xmax)
            axes[i, j].set_xticks(np.arange(xmin, xmax + 0.5, 0.5))
        if 'ylim' in kwargs.keys():
            assert len(kwargs['ylim']) == 2, "x should be of the form (xmin, xmax)"
            ylim = kwargs['ylim']
            ymin = ylim[0]
            ymax = ylim[1]
            axes[i, j].set_ylim(ymin, ymax)

        # Create meshgrid
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = st.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)

        # axes[i, j].set_ylim(ymin, 3)
        cfset[i, j] = axes[i, j].contourf(xx, yy, f, cmap='coolwarm', norm=norm)
        cset = axes[i, j].contour(xx, yy, f, colors='k', norm=norm)
        # axes[i, j].clabel(cset, inline=1, fontsize=10)
        axes[i, j].set_xlabel('$%s$ [mag]' % (x_data[i, j][0] + ' - ' + x_data[i, j][1]))
        if j == 1:
            axes[i, j].yaxis.set_visible(False)

    pos1 = axes[1, 1].get_position()
    cbar_ax = fig.add_axes([0.85, pos1.y0, 0.025, 2 * pos1.height + 0.07])
    colorrange = np.array([cfseti.levels[-1] - cfseti.levels[0] for cfseti in cfset.ravel()], dtype=np.float64).reshape((2, 2))
    indices = np.nanargmax(colorrange, axis=0)
    cbar = fig.colorbar(cfset[indices[0], indices[1]], cax=cbar_ax, label='Probability Density')
    # ax.legend()
    fig.text(0.06, 0.5, 'Redshift ($z$)', ha='center', va='center', rotation='vertical')
    fig.suptitle('2D Gaussian Kernel density estimation - {}'.format(label))
    fig.subplots_adjust(right=0.8, wspace=0.15)
    plt.savefig('quasar-color-redshift-density_{}.png'.format(label.lower().replace(' ', '_')), dpi=300, bbox_extra_artist=(cbar,))
    plt.close()


make_density_plot(norm=colors.Normalize(vmin=0, vmax=1.6), label='Linear Scaling', xlim=(-0.5, 1.0), ylim=(0, 3))
make_density_plot(norm=colors.LogNorm(), label='Logarithmic Scaling')