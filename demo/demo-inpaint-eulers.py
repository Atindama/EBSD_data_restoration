#!/bin/python3

import matplotlib.pyplot as plt
from numpy import deg2rad, pi, isnan
from sys import path
path.insert(0, "..") # hack to get module `tvflow` in scope
import tvflow as tv
from tvflow.misc import range_map

e = deg2rad(tv.fileio.read_ctf("sample_23_3_25_02_", missing_phase=0))
#e = e[300:,300:,:]

eshow = e.copy()
eshow[isnan(eshow)] = 0

plt.figure(figsize=(15,7))
plt.subplot(2,3,1); plt.imshow(range_map(eshow, (0, pi))); plt.title("missing")

u = tv.inpaint(e, delta_tolerance=1e-5, on_quats=False, force_max_iters=False)
plt.subplot(2,3,2); plt.imshow(range_map(u, (0, pi))); plt.title("inpainted")


#u1 = tv.orient.clean_discontinuities(u)
#plt.subplot(2,3,3); plt.imshow(range_map(u1, (0, pi))); plt.title("preprocessed")

u2 = tv.orient.fill_isolated_with_median(u)
plt.subplot(2,3,4); plt.imshow(range_map(u2, (0, pi))); plt.title("median_inpainted")


u3 = tv.denoise(u2, weighted=True, beta=0.0005, on_quats=False)
plt.subplot(2,3,5); plt.imshow(range_map(u3, (0, pi))); plt.title("denoised")

u4 = tv.orient.apply_median_filter(u3)
plt.subplot(2,3,6); plt.imshow(range_map(u4, (0, pi))); plt.title("postprocessed")


plt.show()