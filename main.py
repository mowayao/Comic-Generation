
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import MSELoss, BCELoss
import torch
import pickle
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from data import data_folder
import numpy as np
from net import build_net, Hair_Eye_Embedding, discriminator
from utils import Config
import os
import argparse

parser = argparse.ArgumentParser(description='commic generation')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size(default: 64)')
parser.add_argument('--num_workers', type=int, default=4,
                    help='threads for loading data')
parser.add_argument('--epochs', type=int, default=50,
                    help='epochs for train (default: 50)')
parser.add_argument('--training_iterations', type=int, default=300000,
                    help='training iterations (default: 300000)')
parser.add_argument('--test_interval', type=int, default=5000,
                    help='test interval (default: 5000)')
parser.add_argument('--decay_step', type=int, default=30000,
                    help='decay interval (default: 30000)')
parser.add_argument('--seed', type=int, default=1024, metavar='S',
                    help='random seed (default: 1024)')
parser.add_argument('--visdom', type=bool, default=True,
                    help='visdom')
parser.add_argument('--phase', type=str, default="train",
                    help="phase:train or test")
parser.add_argument('--data_root', type=str, default="data/faces")
parser.add_argument('--lr', type=float, default=0.0002)
args = parser.parse_args()

if args.visdom:
	import visdom

	viz = visdom.Visdom(port=7778)

	epoch_lot_train_dis_loss = viz.line(
		X=torch.zeros((1, )).cpu(),
		Y=torch.zeros((1, )).cpu(),
		opts=dict(
			xlabel='Iterations',
			ylabel='Loss',
			title='Discriminator Training Loss',
			legend=['Loss']
		)
	)
	epoch_lot_train_gen_loss = viz.line(
		X=torch.zeros((1,)).cpu(),
		Y=torch.zeros((1,)).cpu(),
		opts=dict(
			xlabel='Iterations',
			ylabel='Loss',
			title='Generator Training Loss',
			legend=['Loss']
		)
	)

	generated = viz.images(
		np.ones((args.batch_size, 3, Config.img_dim, Config.img_dim)),
		opts=dict(title='Generated')
	)

	gt = viz.images(
		np.ones((args.batch_size, 3, Config.img_dim, Config.img_dim)),
		opts=dict(title='Original')
	)

def preprocess(data_root):
	tag_words = np.load("my_tags_word.npy")
	hair_dict = pickle.load(open("hair_dict", "rb"))
	eyes_dict = pickle.load(open("eyes_dict", "rb"))
	tag_imgs = []
	ids = map(lambda x: x[0], tag_words)
	tags = map(lambda x: (hair_dict[x[1]], eyes_dict[x[2]]), tag_words)
	for ID in ids:
		tag_imgs.append(os.path.join(data_root, ID+".jpg"))

	return tag_imgs, tags

tag_imgs, tags = preprocess(args.data_root)
train_folder = data_folder(tag_imgs, tags, Config.hair_dim, Config.eyes_dim)
train_loader = DataLoader(train_folder, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, drop_last=True)

gen = build_net("DCGAN", nz=500, nc=3, ngf=64, n_extra_layers_g=7).cuda()
embed = Hair_Eye_Embedding(hair_dim = Config.hair_dim, eyes_dim = Config.eyes_dim, embedding_dim = Config.embedding_dim)
dis = discriminator(3, 64).cuda()

bce_loss = BCELoss()
mse_loss = MSELoss()
G_optimizer = optim.Adam(gen.parameters(), betas=(0.5, 0.999), lr=args.lr)
D_optimizer = optim.Adam(dis.parameters(), betas=(0.5, 0.999), lr=args.lr*0.5)
embed_optimizer = optim.Adam(embed.parameters(), lr=args.lr*0.2, betas=(0.5, 0.999))


def conv_edge(img_batch):

	filters = Variable(torch.FloatTensor([[[[1, -1]]]])).cuda()
	return torch.sum(torch.mean(torch.abs(F.conv2d(img_batch[:,0,:,:].unsqueeze(dim=1), filters)), dim=0)) + \
	       torch.sum(torch.mean(torch.abs(F.conv2d(img_batch[:,1,:,:].unsqueeze(dim=1), filters)), dim=0))+ \
			torch.sum(torch.mean(torch.abs(F.conv2d(img_batch[:, 2, :, :].unsqueeze(dim=1), filters)), dim=0))

	#conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)


flag = True

total_iter = 0

sum_d_train_loss = 0
sum_g_train_loss = 0
for epoch in xrange(args.epochs):
	for iteration, (img_batch, hair_batch, eye_batch) in enumerate(train_loader):

		dis.zero_grad()
		embed.zero_grad()

		img_batch = Variable(img_batch).cuda()
		hair_batch = Variable(hair_batch)
		eye_batch = Variable(eye_batch)

		y_real_batch = Variable(torch.ones(args.batch_size)).cuda()
		y_fake_batch = Variable(torch.zeros(args.batch_size)).cuda()
		d_real_res = dis(img_batch)
		D_real_loss = bce_loss(d_real_res, y_real_batch)

		hair_embed_batch, eye_embed_batch = embed(hair_batch, eye_batch)
		hair_embed_batch, eye_embed_batch = hair_embed_batch.cuda(), eye_embed_batch.cuda()
		noise = Variable(torch.randn((args.batch_size, 100))).cuda()

		z = torch.cat((hair_embed_batch, eye_embed_batch, noise), dim=1)

		g_res, g_z = gen(z)

		d_fake_res = dis(g_res.detach())
		D_fake_loss = bce_loss(d_fake_res, y_fake_batch)
		D_train_loss = D_real_loss + D_fake_loss
		D_train_loss.backward()
		D_optimizer.step()


		gen.zero_grad()

		D_res = dis(g_res)

		G_train_loss = bce_loss(D_res, y_real_batch) + mse_loss(g_z, z.detach()) #- 0.001*conv_edge(g_res)


		G_train_loss.backward()
		G_optimizer.step()
		embed_optimizer.step()
		print "Iterations: {}\t discriminator loss:{} \t generator loss:{} \t ".format(iteration + 1,


		                                                                             D_train_loss.cpu().data.numpy()[0],
		                                                                             G_train_loss.cpu().data.numpy()[0])
		sum_g_train_loss += G_train_loss.cpu().data[0]
		sum_d_train_loss += D_train_loss.cpu().data[0]
		if args.visdom:
			if (iteration+1)%20==0:
				viz.line(
					X=torch.ones((1,)).cpu() * total_iter,
					Y=torch.Tensor([D_train_loss.cpu().data[0]]).unsqueeze(0).squeeze(1).cpu(),
					win=epoch_lot_train_dis_loss,
					update='append'
				)
				viz.line(
					X=torch.ones((1,)).cpu() * total_iter,
					Y=torch.Tensor([G_train_loss.cpu().data[0]]).unsqueeze(0).squeeze(1).cpu(),
					win=epoch_lot_train_gen_loss,
					update='append'
				)
				sum_g_train_loss = 0
				sum_d_train_loss = 0
				total_iter += 1
			if iteration == 0:
				viz.images(
					255. * img_batch.cpu().data.numpy(),
					win=generated
				)

				viz.images(
					255. * g_res.cpu().data.numpy(),
					win=gt
				)