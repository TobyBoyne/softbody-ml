from physics.body import Body
import matplotlib.pyplot as plt
import numpy as np

# from scipy.spatial import ConvexHull

NUM_POINTS = 30

def get_mesh(body: Body):
	points = body.particles[body.outer]
	return points



if __name__ == "__main__":
	b = Body(8)
	b.run()
	mesh = get_mesh(b)
	plt.plot(mesh[:, 0], mesh[:, 1], 'o')
	plt.show()