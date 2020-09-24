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


def get_mesh(body: Body):
	"""Return indexes of evenly spaces particles on the outside of the body"""
	return body.outer


def points_to_grid(points: np.ndarray, grid_radius):
	"""Convert a set of points (vertices of a polygon) to a mask"""
	SCALE = 1
	RES = 0.5
	size = grid_radius * SCALE + 1
	ygrid, xgrid = np.mgrid[-size:size:RES, -size:size:RES]
	xypix = np.vstack((xgrid.ravel(), ygrid.ravel())).T

	poly = Polygon(points, closed=True)
	mask = poly.contains_points(xypix)
	return mask.reshape(ygrid.shape)


def body_to_grid(body: Body):
	"""Find a polygon that encloses the shape, then """
	mesh = get_mesh(body)
	points = body.particles[mesh]
	mask = points_to_grid(points, body.R)

	plt.imshow(mask)
	plt.show()
	return mask


if __name__ == "__main__":

	b = Body(8)

	grid = body_to_grid(b)