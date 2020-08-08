import numpy as np

U = np.array([1, 0])
ROT = 0.5 * np.array([[1, - np.sqrt(3)], [np.sqrt(3), 1]])
DISTS = np.array([np.dot(ROT,U), np.dot(ROT@ROT, U)])
print(DISTS)
D = 1

def hex_grid_distance(grid_pos):
	"""Converts a hex-coordinate position to cartesian distance"""
	cartesian = np.dot(grid_pos, D * DISTS)
	print(cartesian)
	return np.linalg.norm(cartesian)


class Body:
	def __init__(self, R):
		self.R = R
		self.particles = []
		self.create_particles()

	def create_particles(self):
		"""Create a hex grid of particles"""
		start_part = Particle(np.array([0, 0]))
		sub_moves = np.array([[1, 0], [0, 1], [1, -1]])
		moves = np.concatenate((sub_moves, -sub_moves))
		grid = {start_part.pos_tuple(): start_part}
		cur_parts = [start_part]
		while cur_parts:
			print(grid)
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



class Particle:
	def __init__(self, grid_pos):
		self.pos = grid_pos
		self.connections = []

	def pos_tuple(self):
		return tuple(self.pos)



if __name__ == "__main__":
	body = Body(3)