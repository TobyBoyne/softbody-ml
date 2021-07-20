import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

import physics.body

INTERVAL = int(1000 / 30)

class Animator(FuncAnimation):
	def __init__(self, fig, ax, body_history, wall_group):
		self.body_history = body_history
		self.wall_history = wall_group.history
		self.markers, = ax.plot([], [], marker='o', lw=0)
		self.walls = [ax.add_patch(Polygon(wall.pts, color='grey')) for wall in wall_group.walls]
		kwargs = {
			'init_func': self.init_func,
			'frames': len(body_history),
			'interval': INTERVAL,
			'blit': True
		}
		super().__init__(fig, self.animate, **kwargs)

	def init_func(self):
		points = self.body_history[0]
		self.markers.set_data(points[:, 0], points[:, 1])

		for i, points in enumerate(self.wall_history[0]):
			self.walls[i].xy = points

		return [self.markers] + self.walls

	def animate(self, i):
		# points = self.body_history[i]
		# xs, ys = points[:, 0], points[:, 1]
		# self.markers.set_data(xs, ys)
		# return [self.markers]

		points = self.body_history[i]
		self.markers.set_data(points[:, 0], points[:, 1])

		for i, points in enumerate(self.wall_history[i]):
			self.walls[i].xy = points

		return [self.markers] + self.walls


if __name__ == '__main__':
	body = physics.body.Body(8)
	body.calculate_forces()

	fig, ax = plt.subplots(figsize=(8, 8))
	ax.set_xlim((-10, 10))
	ax.set_ylim((-10, 10))
	anim = Animator(fig, ax, body.history, physics.body.WALL_GROUP)

	plt.show()
