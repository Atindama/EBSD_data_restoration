#!/bin/python3

from matplotlib.pyplot import show
from numpy import deg2rad
from sys import path
path.insert(0, "..") # hack to get module `tvflow` in scope
import tvflow as tv

e = deg2rad(tv.fileio.read_ctf("missing.ctf", missing_phase=0))
q = tv.orient.eulers_to_quats(e, "xyz")

tv.display.display_quats(q, newaxtitle="original")

u = tv.inpaint(q, delta_tolerance=1e-5)

tv.display.display_quats(u, newaxtitle="inpainted")

show()