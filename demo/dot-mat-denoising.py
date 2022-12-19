# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 21:07:57 2021

@author: 13152
"""

from sys import path
path.insert(0, "..") # hack to get module `tvflow` in scope
import tvflow as tv
from numpy import pi, mean, deg2rad, isnan, zeros_like, nan
import matplotlib.pyplot as plt
from tvflow.misc import range_map


clean = deg2rad(tv.fileio.read_mat('e_clean','clean'))
noise = deg2rad(tv.fileio.read_mat('e_noisy','noisy')); #noise = noise
quad = deg2rad(tv.fileio.read_mat('e_q','e_q')); # a .mat file restored using Mtex's half-quadratic filter



"""denoise"""
prep = tv.orient.clean_discontinuities(noise.copy())
prepped = tv.orient.fill_isolated_with_median(prep,5)

denoised_wtv = tv.denoise(prepped, weighted=True, beta=0.001, on_quats=False, max_iters = 8000)


plt.figure(figsize=(10,10)); plt.imshow(clean/(1*pi)); plt.axis('off');plt.savefig('clean.png',bbox_inches='tight',pad_inches=0)
plt.figure(figsize=(10,10)); plt.imshow(noise/(1*pi)); plt.axis('off');plt.savefig('noisy.png',bbox_inches='tight',pad_inches=0)
plt.figure(figsize=(10,10)); plt.imshow(quad/(1*pi)); plt.axis('off');plt.savefig('quad.png',bbox_inches='tight',pad_inches=0)
plt.figure(figsize=(10,10)); plt.imshow(denoised_wtv/(1*pi)); plt.axis('off');plt.savefig('wTV.png',bbox_inches='tight',pad_inches=0)


print('the l2 error of the noisy image is ', tv.orient.mean_l2_error_per_pixel(clean,noise))
print('the l2 error of the half-quadratic denoising is ', tv.orient.mean_l2_error_per_pixel(clean,quad))
print('the l2 error of the wtv denoising is ', tv.orient.mean_l2_error_per_pixel(clean, denoised_wtv))


"saving output files as ctf"
# tv.fileio.save_ang_data_as_ctf('wtv.ctf',denoised_wtv)
# tv.fileio.save_ang_data_as_ctf('quad.ctf',quad)
# tv.fileio.save_ang_data_as_ctf('noisy.ctf',noise)
# tv.fileio.save_ang_data_as_ctf('clean.ctf',clean)

