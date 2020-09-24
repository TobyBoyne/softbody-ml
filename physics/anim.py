import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import physics.body

INTERVAL = int(1000 / 30)

class Animator(FuncAnimation):
	def __init__(self, fig, ax, history):
		self.history = history
		self.markers, = ax.plot([], [], marker='o', lw=0)
		kwargs = {
			'init_func': self.init_func,
			'frames': len(history),
			'interval': INTERVAL,
			'blit': True
		}
		super().__init__(fig, self.animate, **kwargs)

	def init_func(self):
		points = self.history[0]
		xs, ys = points[:, 0], points[:, 1]
		self.markers.set_data(xs, ys)
		return [self.markers]

	def animate(self, i):
		points = self.history[i]
		xs, ys = points[:, 0], points[:, 1]
		self.markers.set_data(xs, ys)
		return [self.markers]


if __name__ == '__main__':
	body = physics.body.Body(8)
	body.calculate_forces()
	body.draw()

	fig, ax = plt.subplots(figsize=(8, 8))
	ax.set_xlim((-10, 10))
	ax.set_ylim((-10, 10))
	anim = Animator(fig, ax, body.history)
	plt.show()
