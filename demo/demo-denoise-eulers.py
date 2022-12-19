#!/bin/python3


from sys import path
path.insert(0, "..") # hack to get module `tvflow` in scope
import tvflow as tv
from numpy import deg2rad,pi, rad2deg
import matplotlib.pyplot as plt

"""Read noisy ctf file and clean ctf file and restore for comparison
   If clean file is not available, use noisy file as both.
   In that case larger l2 errors indicate better restoration.
"""
clean, noisy, preprocessed,filename = tv.orient.denoising_pipeline('Synthetic_test_noisy.ctf', 'Synthetic_test.ctf', preprocess=True, denoise=True, denoise_type='tvflow', postprocess=False, l2error=True, plots=True)


"""
   To use a different weight function for the edge map other than the one generated from 
   our literature, you may compute and input the array as
   Below, we use the result of our tv flow as weights for weighted tv flow denoising.
"""
e = deg2rad(tv.fileio.read_ctf("noisy.ctf"))

u = tv.denoise(e, weighted=True, beta=0.05, weight_array=tv.denoise(q, weighted=True, beta=0.05))
plt.figure();
plt.imshow(u);
plt.show()

