from physics.body import Body, D
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from tqdm import tqdm

NUM_POINTS = 20

# unit_vec = np.array([0, 1])
# t = 2 * np.pi / NUM_POINTS
# rot_matrix = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
# rays = np.zeros((NUM_POINTS, 2))
#
# for i in range(NUM_POINTS):
# 	rays[i, :]  = unit_vec
# 	unit_vec = np.dot(rot_matrix, unit_vec)
#
#
#
# def get_mesh(body: Body):
# 	outer_points = body.particles[body.outer]
# 	for ray in rays:


"""Find the points that make up the circular mesh"""
def get_angles(points):
	return np.arctan2(points[:, 0], points[:, 1])

def get_mesh(body: Body):
	"""Return indexes of evenly spaces particles on the outside of the body"""
	outer_points = body.particles[body.outer]
	# points = np.sort(outer_points, key=get_angle)
	angles = get_angles(outer_points)
	points = body.outer[np.argsort(angles)]
	idxs = np.linspace(0, len(points)-1, NUM_POINTS, dtype=np.int)
	return points[idxs]


def body_to_grid(body: Body):
	"""Find a polygon that encloses the shape, then """
	SCALE = 1
	RES = 0.5
	size = body.R * SCALE + 1
	ygrid, xgrid = np.mgrid[-size:size:RES, -size:size:RES]
	xypix = np.vstack((xgrid.ravel(), ygrid.ravel())).T

	mesh = get_mesh(body)
	poly = Polygon(b.particles[mesh], closed=True)
	mask = poly.contains_points(xypix)



if __name__ == "__main__":

	b = Body(8)

	grid = body_to_grid(b)



	# plt.show()


	# b = Body(8)
	# # b.run()
	# # x = np.array([-1, 1])
	# # print(get_angles(x))
	# mesh = get_mesh(b)
	# pts = b.particles[mesh]
	# poly = Polygon(pts, closed=True)
	# fig, ax = plt.subplots()
	# ax.set_xlim((-10, 10))
	# ax.set_ylim((-10, 10))
	# ax.add_artist(poly)
	# plt.show()