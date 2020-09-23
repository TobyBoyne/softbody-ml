"""Class to store wall collisions
Each wall is stored as a triangle"""

import numpy as np
from matplotlib.patches import Polygon

from physics.body import Body


# TODO: convert more complex shapes into wall triangles
# speed up intersections using numpy methods

PAIRS = np.array([[0, 1], [1, 2], [2, 0]])

def lines_intersect(L1, L2):
	"""Returns whether two lines intersect"""
	(x1, y1), (x2, y2) = L1
	(x3, y3), (x4, y4) = L2

	denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
	if denom == 0:
		return False
	t = (x1 - x2) * (y3 - y4) - (y1 - y3) * (x3 - x4)
	t /= denom

	u = (x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)
	u /= denom

	return 0 <= t <= 1 and 0 <= u <= 1


def point_to_line(p, L):
	"""Returns the shortest vector from a point p to a line L"""
	a = L[0, :]
	n = L[1, :] - L[0, :]
	n /= np.linalg.norm(n)
	vec = (a - p) - ((a - p) @ n) * n
	return vec

class Wall:
	K = 100
	def __init__(self, pts: np.ndarray):
		self.pts = pts # pts is a (3x2) array
		self.polygon = Polygon(pts, closed=True)


	def outward_vector(self, body: Body):
		"""Find which wall is between the particle and the centre of the body
		Then return the vector perpendicular to that wall"""
		is_inside = self.polygon.contains_points(body.particles)
		force_arr = np.zeros_like(body.particles)
		if not is_inside.any():
			# if no particles are inside the wall, return
			return force_arr
		idxs = np.where(is_inside)
		points = body.particles[idxs]

		for i, point in zip(idxs, points):
			centre_line = np.stack((point, body.centre), axis=-1)
			for line in self.pts[PAIRS]:
				if lines_intersect(centre_line, line):
					break
			else: # nobreak -> no force for this point
				break

			vec = point_to_line(point, line)
			force_arr[i, :] = Wall.K * vec

		return force_arr





if __name__ == '__main__':
	L = np.array([[0., 0.], [100., 10.]])
	p = np.array([20., 40.])
	print(point_to_line(p, L))
