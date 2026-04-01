import numpy as np

import scipy as sp

import matplotlib.pyplot as plt

from itertools import combinations

import matplotlib.cm as cm

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

list_of_markers=[
    # lower nodes
    {"position":[0.0, 0.002706666666657956, 0.005014005867379833], "color":"white"},
    {"position":[-0.009937066770880507, 0.002154044527756695, 0.005014005867379833], "color":"white"},
    {"position":[-0.02005598172531916, 0.00219532818557083, 0.005014005867379835], "color":"white"},
    {"position":[0.00986881339801569, 0.002157886334920429, 0.005014005867379833], "color":"white"},
    {"position":[0.0199910237665205, 0.002169405772122497, 0.005014005867379833], "color":"white"},
    {"position":[-4.534213541115486e-05, 0.009476875077660365, 0.005014005867379833], "color":"white"},
    {"position":[-4.318089313660243e-05, 0.01472370773312667, 0.005014005867379833], "color":"white"},
    {"position":[-4.549057055915243e-05, -0.005464843272612792, 0.005014005867379833], "color":"white"},
    {"position":[-4.331006004236785e-05, -0.01300714018473771, 0.005014005867379833], "color":"white"},
    {"position":[-0.008156171328816792, 0.01063828671865583, 0.005014005867379833], "color":"white"},
    {"position":[-0.01200284109465504, 0.01455883488091643, 0.005014005867379833], "color":"white"},
    {"position":[0.006604393387532548, -0.004133816343724023, 0.005014005867379833], "color":"white"},
    {"position":[0.01193196647380104, -0.009375813586500564, 0.005014005867379833], "color":"white"},
    {"position":[-0.00664049659193075, -0.004079153521379328, 0.005014005867379833], "color":"white"},
    {"position":[-0.01193276957183655, -0.009288741429231317, 0.005014005867379833], "color":"white"},
    {"position":[0.008056372734255662, 0.01066651450660219, 0.005014005867379833], "color":"white"},
    {"position":[0.01196729402242558, 0.01461209246369756, 0.005014005867379833], "color":"white"},
    # outer ring nodes lower
    {"position":[0.01435, 0.0170, 0.005014005867379833], "color":"yellow 5"},
    {"position":[-0.0143147, -0.0116766, 0.005014005867379833], "color":"yellow 5"},
    {"position":[0.0143127, -0.011763717, 0.005014005867379833], "color":"yellow 5"},
    {"position":[-0.014385546, 0.016946737, 0.005014005867379833], "color":"yellow 5"},
    {"position":[-0.08e-3, -0.016395047, 0.005014005867379833], "color":"yellow 5"},
    {"position":[-0.08e-3, 0.017, 0.005014005867379833], "color":"yellow 5"},
    {"position":[0.02442, 0.0022, 0.005014005867379833], "color":"yellow 5"},
    {"position":[-0.02442, 0.0022, 0.005014005867379833], "color":"yellow 5"},
    # upper points
    {"position":[0.0, 0.002706666666657956, 0.008985994132581825], "color":"red 5"},
    {"position":[-0.009937066770880512, 0.002154044527756696, 0.008985994132581825], "color":"red 5"},
    {"position":[-0.02005598172531915, 0.002195328185570831, 0.008985994132581832], "color":"red 5"},
    {"position":[0.009868813398015692, 0.002157886334920429, 0.008985994132581825], "color":"red 5"},
    {"position":[0.0199910237665205, 0.002169405772122496, 0.008985994132581825], "color":"red 5"},
    {"position":[-4.534213541115486e-05, 0.009476875077660367, 0.008985994132581825], "color":"red 5"},
    {"position":[-4.318089313660243e-05, 0.01472370773312667, 0.008985994132581825], "color":"red 5"},
    {"position":[-4.549057055915243e-05, -0.005464843272612793, 0.008985994132581825], "color":"red 5"},
    {"position":[-4.331006004236786e-05, -0.01300714018473771, 0.008985994132581825], "color":"red 5"},
    {"position":[-0.008156171328816792, 0.01063828671865583, 0.008985994132581825], "color":"red 5"},
    {"position":[-0.01200284109465504, 0.01455883488091643, 0.008985994132581825], "color":"red 5"},
    {"position":[0.006604393387532548, -0.004133816343724024, 0.008985994132581825], "color":"red 5"},
    {"position":[0.01193196647380104, -0.009375813586500565, 0.008985994132581825], "color":"red 5"},
    {"position":[-0.00664049659193075, -0.004079153521379328, 0.008985994132581825], "color":"red 5"},
    {"position":[-0.01193276957183655, -0.009288741429231317, 0.008985994132581825], "color":"red 5"},
    {"position":[0.008056372734255662, 0.01066651450660219, 0.008985994132581825], "color":"red 5"},
    {"position":[0.01196729402242558, 0.01461209246369756, 0.008985994132581825], "color":"red 5"},
    # outer ring nodes upper
    {"position":[0.01435, 0.0170, 0.008985994132581825], "color":"yellow 5"},
    {"position":[-0.0143147, -0.0116766, 0.008985994132581825], "color":"yellow 5"},
    {"position":[0.0143127, -0.011763717, 0.008985994132581825], "color":"yellow 5"},
    {"position":[-0.014385546, 0.016946737, 0.008985994132581825], "color":"yellow 5"},
    {"position":[-0.08e-3, -0.016395047, 0.008985994132581825], "color":"yellow 5"},
    {"position":[-0.08e-3, 0.017, 0.008985994132581825], "color":"yellow 5"},
    {"position":[0.02442, 0.0022, 0.008985994132581825], "color":"yellow 5"},
    {"position":[-0.02442, 0.0022, 0.008985994132581825], "color":"yellow 5"}]

points = np.array([dictionary["position"] for dictionary in list_of_markers])

delaunay_class = sp.spatial.Delaunay(points)

print("The simplices are:\n"+str(delaunay_class.simplices))

# Creates a 3D plot

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

# Plots the points

ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=50)

# Labels points

for i, p in enumerate(points):

    ax.text(p[0], p[1], p[2], f'{i}', fontsize=12)

# Plots edges of each simplex (tetrahedron)

for simplex in delaunay_class.simplices:
    for i, j in combinations(simplex, 2):
        ax.plot(
            [points[i, 0], points[j, 0]],
            [points[i, 1], points[j, 1]],
            [points[i, 2], points[j, 2]],
            'k-'
        )

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Delaunay simplices')

plt.show()

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Different colors for each tetra
n_tets = len(delaunay_class.simplices)
cmap = cm.get_cmap('viridis', n_tets)   # try: 'plasma', 'jet', 'tab10', 'turbo'

# ============================================================
# Plot each tetrahedron
# ============================================================

for i, simplex in enumerate(delaunay_class.simplices):
    
    tet_points = points[simplex]

    # Faces of a tetrahedron (4 triangular faces)
    faces = [
        [tet_points[0], tet_points[1], tet_points[2]],
        [tet_points[0], tet_points[1], tet_points[3]],
        [tet_points[0], tet_points[2], tet_points[3]],
        [tet_points[1], tet_points[2], tet_points[3]]
    ]

    poly = Poly3DCollection(
        faces,
        alpha=0.25,                 # transparency
        facecolor=cmap(i),
        edgecolor='k',
        linewidth=1.0
    )

    ax.add_collection3d(poly)

    # Optional: mark tetra centroid with its index
    centroid = tet_points.mean(axis=0)
    ax.text(
        centroid[0], centroid[1], centroid[2],
        f"T{i}",
        fontsize=12,
        color='k'
    )

# ============================================================
# Plot points
# ============================================================

ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='black', s=60)

for i, p in enumerate(points):
    ax.text(p[0], p[1], p[2], f'  {i}', fontsize=12, color='black')

# ============================================================
# Axis labels and formatting
# ============================================================

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Delaunay tetrahedra')

# Equal aspect ratio
xlim = [points[:,0].min(), points[:,0].max()]
ylim = [points[:,1].min(), points[:,1].max()]
zlim = [points[:,2].min(), points[:,2].max()]

max_range = max(
    xlim[1] - xlim[0],
    ylim[1] - ylim[0],
    zlim[1] - zlim[0]
) / 2.0

mid_x = np.mean(xlim)
mid_y = np.mean(ylim)
mid_z = np.mean(zlim)

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.tight_layout()
plt.show()