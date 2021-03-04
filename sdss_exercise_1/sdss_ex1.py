from astropy.io import fits
import glob
from PIL import Image
import matplotlib.pyplot as plt
import os

if not os.path.exists('combined_images'):
    os.mkdir('combined_images')

fits_files = glob.glob('*.fits')
jfifs = glob.glob('*.jfif')
png = glob.glob('*.png')
others = list((jfifs[0], jfifs[1], png[0], jfifs[2]))

fits_images = [fits.getdata(fitsfile) for fitsfile in fits_files]
other_images = [Image.open(other) for other in others]

for i in range(len(fits_images)):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4))
    ax1.imshow(fits_images[i])
    ax2.imshow(other_images[i])
    fig.suptitle(fits_files[i].replace('.fits', ''))
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    plt.savefig('combined_images' + os.sep + fits_files[i].replace('.fits', '.png'), dpi=300)
