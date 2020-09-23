import torch
import torch.nn as nn
import torch.nn.functional as F



# TODO: ensure works smoothly with different image sizes

L = 6
FEATURES = 100

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.encoder = Encoder()
		self.decoder = Decoder()

	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		self.net = nn.Sequential(
			nn.Conv2d(1, 6, 5),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			nn.Conv2d(6, 10, 5),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),

			nn.Flatten(),
			nn.Linear(10 * L * L, FEATURES),
			nn.ReLU()
		)

	def forward(self, x):
		return self.net(x)

class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()
		self.upsample = nn.Upsample(scale_factor=2)
		self.conv1 = nn.ConvTranspose2d(10, 6, 5)
		self.conv2 = nn.ConvTranspose2d(6, 1, 5)
		self.fc = nn.Linear(FEATURES, 10 * L * L)

	def forward(self, x):
		x = F.relu(self.fc(x))
		x = x.view(-1, 10, L, L)
		x = F.relu(self.conv1(self.upsample(x)))
		x = F.relu(self.conv2(self.upsample(x)))
		return x

if __name__ == '__main__':
	encoder = Encoder()
	x_in = torch.randn(1, 1, 36, 36)
	x_mid = encoder.forward(x_in)
	print(x_mid)


	x_code = torch.randn(1, 10)
	decoder = Decoder()
	out = decoder(x_mid)

	loss_fn = nn.MSELoss()
	print(x_in.shape, out.shape)
	print(loss_fn(x_in, out))

