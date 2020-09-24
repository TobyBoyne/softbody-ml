"""Class to store wall collisions
Each wall is stored as a triangle"""

import numpy as np
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt

from physics.body import Body


# TODO: convert more complex shapes into wall triangles
# speed up intersections using numpy methods

PAIRS = np.array([[0, 1], [1, 2], [2, 0]])

def lines_intersect(L1, L2):
	"""Returns whether two lines intersect
	For use in wall.outward_vector to ensure that particle moves towards rest of body
	Currently not being used"""
	(x1, y1), (x2, y2) = L1
	(x3, y3), (x4, y4) = L2

	denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
	if denom == 0:
		return False
	t = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
	t /= denom

	u = - ((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3))
	u /= denom
	return 0 <= t <= 1 and 0 <= u <= 1


def point_to_line(p, L):
	"""Returns the shortest vector from a point p to a line L"""
	a = L[0, :]
	n = L[1, :] - L[0, :]
	if (magnitude := np.linalg.norm(n)) == 0:
		return np.zeros(2)
	n /= magnitude
	vec = (a - p) - ((a - p) @ n) * n
	return vec

class Wall:
	K = 10
	def __init__(self, pts: np.ndarray):
		self.pts = pts # pts is a (3x2) array
		self.polygon = Polygon(pts, closed=True)

	def contains_points(self, points):
		transform = self.polygon.get_transform()
		trans_points = transform.transform(points)
		return self.polygon.contains_points(trans_points)

	def outward_vector(self, body: Body):
		"""Find which wall is between the particle and the centre of the body
		Then return the vector perpendicular to that wall"""
		is_inside = self.contains_points(body.particles)
		force_arr = np.zeros_like(body.particles)
		if not is_inside.any():
			# if no particles are inside the wall, return
			print("HMM")
			return force_arr
		idxs = np.where(is_inside)
		points = body.particles[idxs]

		for i, point in zip(idxs[0], points):
			centre_line = np.stack((point, body.centre), axis=-1)
			vec = None
			for line in self.pts[PAIRS]:
				# if lines_intersect(centre_line, line):
				new_vec = point_to_line(point, line)
				if vec is None or 0 < np.linalg.norm(new_vec) < np.linalg.norm(vec):
					vec = new_vec

			if vec is not None:
				force_arr[i, :] = Wall.K * vec

		return force_arr

	def draw(self, ax: plt.Axes):
		ax.add_artist(self.polygon)





if __name__ == '__main__':
	b = Body(8)
	wall_vertices = np.array([[-1., 3.], [0., -10.], [-10., -10.]])
	wall = Wall(wall_vertices)

	fig,ax = plt.subplots(figsize=(8, 8))

	wall.draw(ax)
	b.draw(fig, ax)

	forces = wall.outward_vector(b)



	for particle, force in zip(b.particles, forces):
		(x0, y0), (dx, dy) = particle, force / wall.K
		if np.linalg.norm(force) > 0:
			ax.plot(x0, y0, marker='o', color='green')
		ax.plot([x0, x0 + dx], [y0, y0 + dy], color='red')


	plt.show()