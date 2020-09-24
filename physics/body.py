import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from time import perf_counter
from tqdm import tqdm

U = np.array([1, 0])
ROT = 0.5 * np.array([[1, - np.sqrt(3)], [np.sqrt(3), 1]])
DISTS = np.array([np.dot(ROT,U), np.dot(ROT@ROT, U)])

D = 1

# physics consts
dt = 0.01
k = 10
eta = 10

# TODO: add force due to viscosity
# sort outer by clockwise direction

def cartesian(grid_pos):
	return np.dot(grid_pos, D * DISTS)

def hex_grid_distance(grid_pos):
	"""Converts a hex-coordinate position to cartesian distance"""
	coords = cartesian(grid_pos)
	return np.linalg.norm(coords)


class Body:
	def __init__(self, R):
		self.R = R
		self.particles, self.connections, outer = self.create_particles()
		self.outer = self.sort_outer(outer)

		# centre stores the (x, y) coordinates of the centre particle
		self.centre = np.array([0., 0.])

		self.velocities = np.zeros_like(self.particles)
		self.forces = np.zeros_like(self.particles)

	def run(self):
		self.calculate_forces()
		self.draw()

	def create_particles(self):
		"""Create a hex grid of particles"""
		sub_moves = np.array([[1, 0], [0, 1], [1, -1]])
		moves = np.concatenate((sub_moves, -sub_moves))

		# outer stores the indices of the particles on the outside of the body
		# defined as particles that have fewer than 6 neighbours
		outer = []

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
				if i not in outer:
					outer.append(i)
		connections = np.array(connections)
		outer = np.array(outer)

		grid_pos = np.array(particles)
		pos = cartesian(grid_pos)
		return pos, connections, outer

	def sort_outer(self, outer):
		def get_angles(points):
			return np.arctan2(points[:, 0], points[:, 1])

		outer_points = self.particles[outer]
		angles = get_angles(outer_points)
		sorted_outer = outer[np.argsort(angles)]
		return sorted_outer


	def calculate_forces(self, wall_perp):
		with tqdm(range(100)) as tqdm_iter:
			for _ in tqdm_iter:
				internal = self.internal_forces()
				applied = self.applied_forces(wall_perp)
				self.forces = internal + applied
				self.step()

	def applied_forces(self, perp):
		"""Applies a force based on wall where `perp` is the perpendicular vector to the wall"""
		d = np.linalg.norm(perp)
		dists = np.dot(self.particles, perp)
		# no force is applied to particles not on the other side of the wall
		dists *= (dists > d)

		# wall stiffness = 0.015
		forces = -0.015 * perp.reshape((-1, 1)) @ dists.reshape((1, -1))
		return forces.T



	def internal_forces(self):
		# get direction vectors for all forces (N, 2, 6)
		stacked_particles = np.expand_dims(self.particles, axis=1).repeat(6, axis=1)
		dir_vecs = self.particles[self.connections] - stacked_particles
		# get amplitudes of direction vectors
		N = self.particles.shape[0]
		amplitudes = np.linalg.norm(dir_vecs, axis=-1).reshape((N, 6, 1))
		# set zero amplitudes to 1, avoid divide-by-zero errors
		amplitudes[np.nonzero(amplitudes == 0)] = 1.
		forces = k * (1 - D / amplitudes) * dir_vecs
		return forces.sum(axis=1)



	def step(self):
		self.velocities += self.forces * dt
		self.particles += self.velocities * dt

		centre = self.particles[0]
		self.centre += centre
		self.particles -= centre



	def draw(self, fig=None, ax=None):
		if fig is None and ax is None:
			fig, ax = plt.subplots(figsize=(8, 8))
		ax.set_xlim((-10, 10))
		ax.set_ylim((-10, 10))
		for p in self.particles:
			circle = plt.Circle(p, D/5, color='b')
			ax.add_artist(circle)





if __name__ == "__main__":
	body = Body(8)
	body.run()