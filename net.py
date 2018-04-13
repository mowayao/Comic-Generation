import torch
import torch.nn as nn
import torch.nn.functional as F

class Hair_Eye_Embedding(nn.Module):
	def __init__(self, hair_dim, eyes_dim, embedding_dim):
		super(Hair_Eye_Embedding, self).__init__()
		self.hair_embed = nn.Embedding(hair_dim, embedding_dim)
		self.eye_embed = nn.Embedding(eyes_dim, embedding_dim)
	def forward(self, hair, eyes):
		hair = self.hair_embed(hair)
		eyes = self.eye_embed(eyes)
		return hair, eyes
class DCGAN(nn.Module):##generator
	def __init__(self, nz, nc, ngf):
		super(DCGAN, self).__init__()

		self.convs = nn.Sequential(
			# input is Z, going into a convolution
			# state size. nz x 1 x 1
			nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
			nn.BatchNorm2d(ngf * 8, momentum=0.9, eps=1e-5),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ngf*8) x 4 x 4
			nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 4, momentum=0.9, eps=1e-5),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ngf*4) x 8 x 8
			nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 2, momentum=0.9, eps=1e-5),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ngf*2) x 16 x 16
			nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf, momentum=0.9, eps=1e-5),
			nn.LeakyReLU(0.2, inplace=True),
			nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
			nn.Tanh()
			# state size. (ngf) x 32 x 32
		)

	def forward(self, x):
		x = self.convs(x)/2.0 + 0.5
		return x

class discriminator(nn.Module):
	def __init__(self, nc, ndf):
		super(discriminator, self).__init__()
		self.convs = nn.Sequential(
			# input is (nc) x 96 x 96
			nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 32 x 32
			nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 16 x 16
			nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*4) x 8 x 8
			nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 8),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*8) x 4 x 4
			nn.Conv2d(ndf * 8, 1024, 4, 2, 1, bias=False),
			nn.LeakyReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Conv2d(1024, 1024, 3, 2, 1, bias=False),
			nn.LeakyReLU(inplace=True),
			nn.Dropout(0.5),
			# state size. 1024 x 1 x 1
		)
		self.fcs = nn.Sequential(
			nn.Linear(1024, 1024),
			nn.LeakyReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(1024, 1),
			nn.Sigmoid()
		)
	# weight_init

    # forward method
	def forward(self, x):
		x = self.convs(x)
		return self.fcs(x.view(-1, 1024))
class discriminator_wgan(nn.Module):
	def __init__(self, nc, ndf):
		super(discriminator_wgan, self).__init__()
		self.convs = nn.Sequential(
	        # input is (nc) x 96 x 96
	        nn.Conv2d(nc, ndf, 4, 2, 1),
	        nn.LeakyReLU(0.2, inplace=True),
	        # state size. (ndf) x 32 x 32
	        nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
	        nn.BatchNorm2d(ndf * 2),
	        nn.LeakyReLU(0.2, inplace=True),
	        # state size. (ndf*2) x 16 x 16
	        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
	        nn.BatchNorm2d(ndf * 4),
	        nn.LeakyReLU(0.2, inplace=True),
	        # state size. (ndf*4) x 8 x 8
	        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
	        nn.BatchNorm2d(ndf * 8),
	        nn.LeakyReLU(0.2, inplace=True),
	        # state size. (ndf*8) x 4 x 4
	        # state size. 1024 x 1 x 1
        )
		self.combine_conv = nn.Sequential(
			nn.Conv2d(ndf * 8 + 23, ndf * 8, 4, 1, 0),
			nn.BatchNorm2d(ndf * 8),
			nn.LeakyReLU(0.2, inplace=True)
		)
		self.fc = nn.Sequential(
			nn.Linear(ndf * 8, 1),
			nn.Sigmoid()
		)


	def forward(self, x, label):
		b = x.size(0)
		x = self.convs(x)

		label = label.expand(b, 23, x.size()[-1], x.size()[-1])
		x = torch.cat((x, label), dim=1)
		x = self.combine_conv(x)
		return self.fc(x.view(b, -1))
class WGAN(nn.Module):
	def __init__(self):
		super(WGAN, self).__init__()

	def forward(self, x):
		pass



class WGAN_v2(nn.Module):
	def __init__(self):
		super(WGAN_v2, self).__init__()

	def forward(self, x):
		pass



def build_net(model, **kwargs):
	if model == "DCGAN":
		return DCGAN(**kwargs)


