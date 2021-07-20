import numpy as np
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt

from physics.body_to_mesh import points_to_grid

from learning.autoencoder.network import Net

if __name__ == '__main__':
	R = 8

	net = Net()
	optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
	loss_fn = torch.nn.MSELoss()

	with tqdm(range(1000)) as tqdm_iter:
		unit = np.array([1, 0])
		t = np.linspace(0, 2 * np.pi, 50)
		for i in tqdm_iter:
			noise = np.random.random() * np.sin(5*t) + np.random.random() * 0.5 * np.sin(9*t)
			radii = np.expand_dims(R + noise, axis=-1)

			rot = np.array([[np.cos(t), -np.sin(t)],
							[np.sin(t), np.cos(t)]])
			rot_stack = np.rollaxis(rot, axis=-1)
			points = radii * (rot_stack @ unit)

			mask = points_to_grid(points, R)

			optimizer.zero_grad()
			x_in = torch.from_numpy(mask.reshape(1, 1, 36, 36)).type(torch.FloatTensor)
			x_out = net(x_in)
			loss = loss_fn(x_in, x_out)
			loss.backward()
			optimizer.step()

			if i % 100 == 0:
				tqdm_iter.set_description(f"{loss:5f}")


	fig, (ax1, ax2) = plt.subplots(2)
	ax1.imshow(mask)
	x_in = torch.from_numpy(mask.reshape(1, 1, 36, 36)).type(torch.FloatTensor)
	x_out = net(x_in)
	out_mask = x_out.detach().numpy().reshape(36, 36)
	print(out_mask)
	ax2.imshow(out_mask)
	plt.show()


