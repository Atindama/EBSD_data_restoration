from sys import path
path.insert(0, "..") # hack to get module `tvflow` in scope

import numpy as np
import tvflow as tv
import skimage.morphology as morph
import matplotlib.pyplot as plt
from skimage.filters import gaussian, farid
from numpy import pi, deg2rad

"""
we test our denoising with an alternative kernel
"""

filename = "Synthetic_test_noisy.ctf"

e = (2*pi/360) * tv.fileio.read_ctf( filename )

prep = tv.orient.clean_discontinuities(e.copy())
prepped = tv.orient.fill_isolated_with_median(prep,5)


print("Running unweighted TV to compute weight function...")
w = tv.misc.weight_from_TV_solution( e )


print("Done! Now running weighted TV...")
e_wtv_alt = tv.denoise( prepped, weighted=True, on_quats=False, weight_array=w, max_iters = 8000)

print("Displaying weighted TV results.")

plt.figure(figsize=(15,15))
plt.subplot(1,3,2);  plt.imshow( e/2/pi )
plt.subplot(1,3,3);  plt.imshow( e_wtv_alt/2/pi )


clean = deg2rad(tv.fileio.read_ctf('Synthetic_test.ctf'))

plt.subplot(1,3,1);  plt.imshow( clean/2/pi )
plt.show();

print('the l2 error of the noisy image is ', tv.orient.mean_l2_error_per_pixel(clean,e),' with misorientation, ',tv.orient.misorientation_error(clean, e))

print('the l2 error of the wtv denoising is ', tv.orient.mean_l2_error_per_pixel(clean, e_wtv_alt),' with misorientation, ',tv.orient.misorientation_error(clean, e_wtv_alt))





"Using the default weighted tv kernel"
e_wtv = tv.denoise( prepped, weighted=True, on_quats=False, max_iters = 8000)

print("Displaying weighted TV results.")

plt.figure(figsize=(15,15))
plt.subplot(1,3,2);  plt.imshow( e/2/pi )
plt.subplot(1,3,3);  plt.imshow( e_wtv/2/pi )


plt.subplot(1,3,1);  plt.imshow( clean/2/pi )


print('the l2 error of the noisy image is ', tv.orient.mean_l2_error_per_pixel(clean,e),' with misorientation, ',tv.orient.misorientation_error(clean, e))

print('the l2 error of the wtv denoising is ', tv.orient.mean_l2_error_per_pixel(clean, e_wtv),' with misorientation, ',tv.orient.misorientation_error(clean, e_wtv))
