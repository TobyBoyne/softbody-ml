import numpy as np
import matplotlib.pyplot as plt

U = np.array([1, 0])
ROT = 0.5 * np.array([[1, - np.sqrt(3)], [np.sqrt(3), 1]])
DISTS = np.array([np.dot(ROT,U), np.dot(ROT@ROT, U)])

D = 1

def cartesian(grid_pos):
	return np.dot(grid_pos, D * DISTS)

def hex_grid_distance(grid_pos):
	"""Converts a hex-coordinate position to cartesian distance"""
	coords = cartesian(grid_pos)
	return np.linalg.norm(coords)


class Body:
	def __init__(self, R):
		self.R = R
		self.particles = []
		self.create_particles()
		self.draw()

	def create_particles(self):
		"""Create a hex grid of particles"""
		sub_moves = np.array([[1, 0], [0, 1], [1, -1]])
		moves = np.concatenate((sub_moves, -sub_moves))

		start_part = Particle(np.array([0, 0]))
		grid = {start_part.pos_tuple(): start_part}
		cur_parts = [start_part]
		self.particles.append(start_part)
		while cur_parts:
			prev_parts = cur_parts
			cur_parts = []

			for part in prev_parts:
				for move in moves:
					next_pos = part.pos + move

					if hex_grid_distance(next_pos) > self.R:
						continue

					if tuple(next_pos) in grid:
						next_part = grid[tuple(next_pos)]
					else:
						next_part = Particle(next_pos)
						grid[tuple(next_pos)] = next_part
						cur_parts.append(next_part)
						self.particles.append(next_part)
					part.connections.append(next_part)

	def draw(self):
		fig, ax = plt.subplots(figsize=(8, 8))
		ax.set_xlim((-10, 10))
		ax.set_ylim((-10, 10))
		for p in self.particles:
			p.draw(ax)
		plt.show()


class Particle:
	def __init__(self, grid_pos):
		self.pos = grid_pos
		self.connections = []

	def pos_tuple(self):
		return tuple(self.pos)



	def draw(self, ax):
		x, y = np.dot(self.pos, D * DISTS)
		circle = plt.Circle((x, y), D/2, color='b')
		ax.add_artist(circle)




if __name__ == "__main__":
	body = Body(10)