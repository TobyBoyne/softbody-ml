"""Class to store wall collisions
Each wall is stored as a triangle"""

import numpy as np
from matplotlib.patches import Polygon

from physics.body import Body


# TODO: convert more complex shapes into wall triangles

class Wall:
	def __init__(self, pts: np.ndarray):
		self.pts = pts
		self.polygon = Polygon(pts, closed=True)

	def contains_points(self, points):
		return self.polygon.contains_points(points)

	def outward_vector(self, body: Body):
		"""Find which wall is between the particle and the centre of the body
		Then return the vector perpendicular to that wall"""
