"""Class to store wall collisions
Each wall is stored as a triangle"""

import numpy as np
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt

import physics.body


# TODO: speed up by using numpy methods

def point_to_line(p, L):
	"""Returns the shortest vector from a point p to a line L"""
	a = L[0, :]
	n = L[1, :] - L[0, :]
	if (magnitude := np.linalg.norm(n)) == 0:
		return np.zeros(2)
	n /= magnitude
	vec = (a - p) - ((a - p) @ n) * n
	return vec

class WallGroup:
	def __init__(self, *wall_pts):
		self.walls = [Wall(pts) for pts in wall_pts]
		self.history = []

	def record_history(self):
		histories = [wall.pts.copy() for wall in self.walls]
		self.history.append(histories)

	def move(self, move_vec):
		for wall in self.walls:
			wall.move(move_vec)

	def get_forces(self, body: 'physics.body.Body'):
		forces = []
		for wall in self.walls:
			forces.append(wall.get_forces(body))

		return np.sum(forces, axis=0)


class Wall:
	K = 1000
	def __init__(self, pts: np.ndarray):
		self.pts = pts # pts is a (Nx2) array
		self.polygon = Polygon(pts, closed=True, color='grey')
		self.pairs = self.get_pairs()

	def get_pairs(self):
		first = np.arange(self.pts.shape[0])
		second = np.roll(first, -1)
		pairs = np.stack((first, second), axis=-1)
		return pairs

	def move(self, move_vec):
		self.pts += move_vec
		self.polygon.xy = self.pts

	def contains_points(self, points):
		transform = self.polygon.get_transform()
		trans_points = transform.transform(points)
		return self.polygon.contains_points(trans_points)

	def get_forces(self, body: 'physics.body.Body'):
		"""Find the forces exerted by the wall on all of the particles in the body"""
		is_inside = self.contains_points(body.particles)
		force_arr = np.zeros_like(body.particles)
		if not is_inside.any():
			# if no particles are inside the wall, return
			return force_arr
		idxs = np.where(is_inside)
		points = body.particles[idxs]

		for i, point in zip(idxs[0], points):
			centre_line = np.stack((point, body.centre), axis=-1)
			vec = None
			for line in self.pts[self.pairs]:
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
	b = physics.body.Body(8)
	wall_vertices = np.array([[-1., 3.], [0., -10.], [-10., -10.]])
	wall = Wall(wall_vertices)
	wall.move(np.array([4., -5.]))
	fig,ax = plt.subplots(figsize=(8, 8))

	wall.draw(ax)
	b.draw(fig, ax)

	forces = wall.get_forces(b)



	for particle, force in zip(b.particles, forces):
		(x0, y0), (dx, dy) = particle, force / wall.K
		if np.linalg.norm(force) > 0:
			ax.plot(x0, y0, marker='o', color='green')
		ax.plot([x0, x0 + dx], [y0, y0 + dy], color='red')


	plt.show()