"""

# Copyright (c) Mindseye Biomedical LLC. All rights reserved.
# Distributed under the (new) CC BY-NC-SA 4.0 License. See LICENSE.txt for more info.

	Read in a data file and plot it using an algorithm. 

"""
from __future__ import division, absolute_import, print_function
import numpy as np
import matplotlib.pyplot as plt
import OpenEIT.dashboard
import OpenEIT.reconstruction 
from OpenEIT.reconstruction.pyeit.mesh import set_perm
from OpenEIT.reconstruction.pyeit.eit.fem import Forward


def parse_line(line):
    try:
        _, data = line.split(":", 1)
    except ValueError:
        return None
    items = []
    for item in data.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            items.append(float(item))
        except ValueError:
            return None
    return np.array(items)

n_el = 16
hsmall = 0.05

""" Load Data: select a file you have created by simdata.py, or recorded through the dashboard """
text_file = open("200928lungs_working3.txt", "r")
lines       = text_file.readlines()
print ("length lines: ",len(lines))
# This is the baseline image.
f0          = parse_line(lines[0]).tolist()
# this is the new difference image. 
f1          = parse_line(lines[1]).tolist()
f2          = parse_line(lines[2]).tolist()
f3          = parse_line(lines[3]).tolist()
f4          = parse_line(lines[4]).tolist()
f5          = parse_line(lines[5]).tolist()

f0=[0] * len(f1)
""" Select one of the three methods of EIT tomographic reconstruction, Gauss-Newton(Jacobian), GREIT, or Back Projection(BP)"""
# This is the Gauss Newton Method for tomographic reconstruction. 
g = OpenEIT.reconstruction.JacReconstruction(n_el=n_el,h0=hsmall)
# Note: Greit method uses a different mesh, so the plot code will be different.
# g = OpenEIT.reconstruction.GreitReconstruction(n_el=n_el)
# 
#g = OpenEIT.reconstruction.BpReconstruction(n_el=n_el)

#mesh_obj = g.mesh_obj
el_pos = g.el_pos
ex_mat = g.ex_mat
pts     = g.mesh_obj['node']
tri = g.mesh_obj['element']
x   = pts[:, 0]
y   = pts[:, 1]

mesh_new = set_perm(g.mesh_obj, anomaly=None, background=2.)
perm = mesh_new['perm']


# plot the mesh
fig, ax = plt.subplots(figsize=(6, 4))
ax.triplot(pts[:, 0], pts[:, 1], tri, linewidth=1)
ax.plot(pts[el_pos, 0], pts[el_pos, 1], 'ro')
ax.axis('equal')
ax.axis([-1.2, 1.2, -1.2, 1.2])
ax.set_xlabel('x')
ax.set_ylabel('y')
title_src = 'number of triangles = ' + str(np.size(tri, 0)) + ', ' + \
            'number of nodes = ' + str(np.size(pts, 0))
ax.set_title(title_src)
plt.show()


# show input permittivitiy
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.tripcolor(pts[:, 0], pts[:, 1], tri,
                  np.real(perm), shading='flat', cmap=plt.cm.viridis)
fig.colorbar(im)
ax.axis('equal')
ax.set_title(r'$\Delta$ Conductivities')
plt.show()

## calculate forward for showing
## draw equi-potential lines
#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#vf = np.linspace(min(g.eit), max(g.eit), 32)
#ax1.tricontour(x, y, tri, g.eit, vf, cmap=plt.cm.viridis)
#plt.show()

data_baseline = f0
print ('f0',len(f0),len(f1))
g.update_reference(data_baseline)
# set the baseline. 
baseline = g.eit_reconstruction(f0)
# do the reconstruction. 
difference_image = g.eit_reconstruction(f4)


""" Uncomment the below code if you wish to plot the Jacobian(Gauss-Newton) or Back Projection output. Also, please look at the pyEIT documentation on how to optimize and tune the algorithms. A little tuning goes a long way! """
# JAC OR BP RECONSTRUCTION SHOW # 
fig = plt.figure()
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.tripcolor(x,y, tri, difference_image,
                  shading='flat', cmap=plt.cm.gnuplot)
ax.plot(x[el_pos], y[el_pos], 'ro')
for i, e in enumerate(el_pos):
    ax.text(x[e], y[e], str(i+1), size=12)
ax.axis('equal')
fig.colorbar(im)
#fig.savefig('figs/demo_2circ.png', dpi=96)
plt.show()
plt.close(fig)


""" Uncomment the below code if you wish to plot the GREIT output. Also, please look at the pyEIT documentation on how to optimize and tune the algorithms. A little tuning goes a long way! """
# GREIT RECONSTRUCION IMAGE SHOW # 
# new     = difference_image[np.logical_not(np.isnan(difference_image))]
# flat    = new.flatten()
# av      = np.median(flat)
# total   = []
# for i in range(32):
#     for j in range(32):
#         if difference_image[i,j] < -5000: 
#             difference_image[i,j] = av

# print ('image shape: ',difference_image.shape)
# fig, ax = plt.subplots(figsize=(6, 4))
# #rotated = np.rot90(image, 1)
# im = ax.imshow(difference_image, interpolation='none', cmap=plt.cm.rainbow)
# fig.colorbar(im)
# ax.axis('equal')
# ax.set_title(r'$\Delta$ Conductivity Map of Lungs')
# fig.set_size_inches(6, 4)
# # fig.savefig('../figs/demo_greit.png', dpi=96)
# plt.show()


