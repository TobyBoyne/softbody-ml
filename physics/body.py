import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from tqdm import tqdm

U = np.array([1, 0])
ROT = 0.5 * np.array([[1, - np.sqrt(3)], [np.sqrt(3), 1]])
DISTS = np.array([np.dot(ROT,U), np.dot(ROT@ROT, U)])

D = 1

# physics consts
dt = 0.01
k = 10

def cartesian(grid_pos):
	return np.dot(grid_pos, D * DISTS)

def hex_grid_distance(grid_pos):
	"""Converts a hex-coordinate position to cartesian distance"""
	coords = cartesian(grid_pos)
	return np.linalg.norm(coords)


class Body:
	def __init__(self, R):
		self.R = R
		self.particles, self.connections = self.create_particles()

		self.velocities = np.zeros_like(self.particles)
		self.forces = np.zeros_like(self.particles)
		self.apply_forces()

		self.draw()

	def create_particles(self):
		"""Create a hex grid of particles"""
		sub_moves = np.array([[1, 0], [0, 1], [1, -1]])
		moves = np.concatenate((sub_moves, -sub_moves))

		grid_to_idx = {(0, 0): 0}
		cur_parts = [0]
		particles = [[0, 0]]
		connections = [[]]
		while cur_parts:
			# for each newly created particle, create a new particle in each of the 6 hex directions
			prev_parts = cur_parts
			cur_parts = []

			for idx in prev_parts:
				grid_pos = particles[idx]
				for move in moves:
					next_pos = grid_pos + move
					if hex_grid_distance(next_pos) > self.R:
						continue

					if tuple(next_pos) in grid_to_idx:
						next_idx = grid_to_idx[tuple(next_pos)]
					else:
						# create a new particle if one does not exist at that grid space
						next_idx = len(particles)
						grid_to_idx[tuple(next_pos)] = next_idx
						connections.append([])
						cur_parts.append(next_idx)
						particles.append(list(next_pos))

					connections[idx].append(next_idx)

		# fill up all connections with reference to self to fill up to six
		# ensures that array has equal length rows
		for i, con in enumerate(connections):
			while len(con) < 6:
				con.append(i)
		connections = np.array(connections)

		grid_pos = np.array(particles)
		pos = cartesian(grid_pos)
		return pos, connections

	def apply_forces(self):
		with tqdm(range(2000)) as tqdm_iter:
			for _ in tqdm_iter:
				self.calc_forces()
				self.step()


	def calc_forces(self):
		# get direction vectors for all forces (N, 2, 6)
		stacked_particles = np.expand_dims(self.particles, axis=1).repeat(6, axis=1)
		dir_vecs = self.particles[self.connections] - stacked_particles
		# get amplitudes of direction vectors
		N = self.particles.shape[0]
		amplitudes = np.linalg.norm(dir_vecs, axis=-1).reshape((N, 6, 1))
		# set zero amplitudes to 1, avoid divide-by-zero errors
		amplitudes[np.nonzero(amplitudes == 0)] = 1.
		forces = k * (1 - D / amplitudes) * dir_vecs
		self.forces = forces.sum(axis=1)

		self.forces[-1] += 10 * np.array([-1., 1.])

	def step(self):
		self.velocities += self.forces * dt
		self.particles += self.velocities * dt


	def draw(self):
		fig, ax = plt.subplots(figsize=(8, 8))
		ax.set_xlim((-10, 10))
		ax.set_ylim((-10, 10))
		for p in self.particles:
			circle = plt.Circle(p, D/2, color='b')
			ax.add_artist(circle)


		# last = self.particles[-1]
		# circle = plt.Circle(tuple(last.pos), D / 3, color='r')
		# ax.add_artist(circle)
		plt.show()





if __name__ == "__main__":
	body = Body(10)