#!/bin/python3

from matplotlib.pyplot import show
from sys import path
path.insert(0, "..") # hack to get module `tvflow` in scope
import tvflow as tv

e = tv.fileio.read_ctf("3D_missing_noisy.ctf", missing_phase=0)
q = tv.orient.eulers_to_quats(e, "xyz")

tv.display.volumetric_displays_quats(
    q, titles=("input, axis 1", "input, axis 2", "input, axis 3")
)

inpainted = tv.inpaint(q, delta_tolerance=1e-5)
u = tv.denoise(inpainted, weighted=True, beta=0.05)

tv.display.volumetric_displays_quats(
    u, titles=("output, axis 1", "output, axis 2", "output, axis 3")
)

show()