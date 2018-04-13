
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import MSELoss, BCELoss, L1Loss
import torch
import torch.nn.functional as F
import pickle
import torch.optim as optim
from torchvision.transforms import ToTensor, Compose, Resize
from torchnet.logger import VisdomPlotLogger, VisdomLogger
import torchnet as tnt
from torch.optim.lr_scheduler import StepLR
from data import data_folder
import numpy as np
from net import build_net, Hair_Eye_Embedding, discriminator_wgan
from utils import Config as cfg
import os
import argparse
import datetime

parser = argparse.ArgumentParser(description='commic generation')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size(default: 64)')
parser.add_argument('--num_workers', type=int, default=4,
                    help='threads for loading data')
parser.add_argument('--epochs', type=int, default=300,
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
parser.add_argument('--visdom_port', type=int, default=7777,
                    help='visdom port')
parser.add_argument('--phase', type=str, default="train",
                    help="phase:train or test")
parser.add_argument('--data_root', type=str, default="/media/mowayao/data/comic_generation/faces")
parser.add_argument('--lr', type=float, default=0.0002)
args = parser.parse_args()

if args.visdom:
	import visdom

	viz = visdom.Visdom(port=args.visdom_port)

	train_dis_loss_logger = VisdomPlotLogger(
		'line', port=args.visdom_port, opts={'title': 'Train Discriminator Loss-{}'.format(datetime.datetime.now())}
	)
	train_gen_loss_logger = VisdomPlotLogger(
		'line', port=args.visdom_port, opts={'title': 'Train Generator Loss-{}'.format(datetime.datetime.now())}
	)

	generated = viz.images(
		np.ones((args.batch_size, 3, cfg.img_dim, cfg.img_dim)),
		opts=dict(title='Generated')
	)

	gt = viz.images(
		np.ones((args.batch_size, 3, cfg.img_dim, cfg.img_dim)),
		opts=dict(title='Original')
	)

def conv_edge(img_batch):
	filters = Variable(torch.FloatTensor([[[1, -1]]])).cuda()
	return torch.sum(torch.mean(torch.abs(F.conv2d(img_batch[:, 0, :, :].unsqueeze(dim=1), filters)), dim=0)) + \
		   torch.sum(torch.mean(torch.abs(F.conv2d(img_batch[:, 1, :, :].unsqueeze(dim=1), filters)), dim=0)) + \
		   torch.sum(torch.mean(torch.abs(F.conv2d(img_batch[:, 2, :, :].unsqueeze(dim=1), filters)), dim=0))

def preprocess(data_root):
	tag_words = np.load("my_tags_word.npy")
	hair_dict = pickle.load(open("hair_dict", "rb"))
	eyes_dict = pickle.load(open("eyes_dict", "rb"))
	tag_imgs = []
	ids = list(map(lambda x: x[0], tag_words))
	tags = list(map(lambda x: (hair_dict[x[1].decode('ascii')], eyes_dict[x[2].decode('ascii')]), tag_words))
	for ID in ids:
		tag_imgs.append(os.path.join(data_root, ID.decode('ascii')+".jpg"))

	return tag_imgs, tags

tag_imgs, tags = preprocess(args.data_root)
transforms = Compose([
			Resize((64, 64)),
			ToTensor()
		])
train_folder = data_folder(tag_imgs, tags, cfg.hair_dim, cfg.eyes_dim, transforms)

train_loader = DataLoader(train_folder, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, drop_last=True)

hair_onehot = torch.zeros(12, 12)
hair_onehot = hair_onehot.scatter_(1, torch.LongTensor(list(range(12))).view(12,1), 1).view(12, 12, 1, 1)

eyes_onehot = torch.zeros(11, 11)
eyes_onehot = eyes_onehot.scatter_(1, torch.LongTensor(list(range(11))).view(11,1), 1).view(11, 11, 1, 1)


gen = build_net("DCGAN", nz=500, nc=3, ngf=64).cuda()
embed = Hair_Eye_Embedding(hair_dim = cfg.hair_dim, eyes_dim = cfg.eyes_dim, embedding_dim = cfg.embedding_dim).cuda()
dis = discriminator_wgan(3, 64).cuda()

train_dis_meter_loss = tnt.meter.AverageValueMeter()
train_gen_meter_loss = tnt.meter.AverageValueMeter()

bce_loss = BCELoss().cuda()
mse_loss = MSELoss().cuda()
l1_loss = L1Loss().cuda()
G_optimizer = optim.Adam(gen.parameters(), betas=(0.5, 0.999), lr=args.lr)
D_optimizer = optim.Adam(dis.parameters(), betas=(0.5, 0.999), lr=args.lr)
#embed_optimizer = optim.Adam(embed.parameters(), lr=args.lr*10, betas=(0.5, 0.999))

G_scheduler = StepLR(optimizer=G_optimizer, step_size=20, gamma=0.5)
D_scheduler = StepLR(optimizer=D_optimizer, step_size=20, gamma=0.5)
total_iter = 0
for epoch in range(args.epochs):
	train_gen_meter_loss.reset()
	train_dis_meter_loss.reset()
	gen.train()
	dis.train()
	num_batch = int(len(train_folder) / args.batch_size)
	G_scheduler.step()
	D_scheduler.step()
	for iteration, (img_batch, hair_batch, eye_batch, fake_hair_batch, fake_eye_batch) in enumerate(train_loader):

		dis.zero_grad()


		img_batch = Variable(img_batch).cuda()

		hair_batch = Variable(hair_batch.long()).cuda()
		eye_batch = Variable(eye_batch.long()).cuda()
		fake_hair_batch = Variable(fake_hair_batch.long()).cuda()
		fake_eye_batch = Variable(fake_eye_batch.long()).cuda()
		real_hair_embed, real_eyes_embed = embed(hair_batch.long(), eye_batch.long())
		fake_hair_embed, fake_eyes_embed = embed(fake_hair_batch.long(), fake_eye_batch.long())

		true_label = torch.cat([real_hair_embed, real_eyes_embed], 1)
		fake_label = torch.cat([fake_hair_embed, fake_eyes_embed], 1)

		y_real_batch = Variable(torch.ones(args.batch_size)).cuda()
		y_fake_batch = Variable(torch.zeros(args.batch_size)).cuda()

		noise = Variable(torch.randn((args.batch_size, 100)).view(-1, 100, 1, 1)).cuda()
		fake_noise = Variable(torch.randn((args.batch_size, 100)).view(-1, 100, 1, 1)).cuda()

		real_z = torch.cat((true_label, noise), dim=1)
		fake_z = torch.cat((fake_label, noise), dim=1)

		g_res = gen(real_z) ##generated image
		fake_g_res = gen(fake_z) ##false generated image

		real_d_res = dis(img_batch, true_label)
		d_loss1 = bce_loss(real_d_res, y_real_batch)


		d_fake_res = dis(g_res.detach(), fake_label) ##generate fake image
		d_loss2 = bce_loss(d_fake_res, y_fake_batch) ###generated fake image and fake babel


		d_false_res = dis(fake_g_res.detach(), fake_label) ##fake
		d_loss3 = bce_loss(d_fake_res, y_fake_batch)


		perm_index = torch.randperm(len(img_batch))
		x_wrong = Variable(img_batch.cpu().data[perm_index]).cuda()
		#x_wrong, y_label_ = Variable(x_wrong.cuda()), Variable(y_label_.cuda())  # x_ : [128, 1, 64, 64]
		d_fake_res1 = dis(x_wrong, true_label)  # (real img, right text): 1
		d_loss4 = bce_loss(d_fake_res1, y_fake_batch)  # wrong picture & real label

		D_train_loss = d_loss1 + d_loss2 + d_loss3 + d_loss4
		D_train_loss.backward()

		D_optimizer.step()
		###
		for _ in range(5):
			gen.zero_grad()
			real_hair_embed, real_eyes_embed = embed(hair_batch.long(), eye_batch.long())
			fake_hair_embed, fake_eyes_embed = embed(fake_hair_batch.long(), fake_eye_batch.long())
			true_label = torch.cat([real_hair_embed, real_eyes_embed], 1)
			fake_label = torch.cat([fake_hair_embed, fake_eyes_embed], 1)
			real_z = torch.cat((true_label, noise), dim=1)
			fake_z = torch.cat((fake_label, noise), dim=1)
			real_g_res = gen(real_z)
			fake_g_res = gen(fake_z)
			D_res = dis(real_g_res, true_label)
			D_fake_res = dis(fake_g_res, fake_label)
			G_train_loss = bce_loss(D_res, y_real_batch) + bce_loss(D_fake_res, y_fake_batch) + mse_loss(real_g_res, img_batch) #+ 0.1*conv_edge(g_res)# + bce_loss(D_res, y_real_batch) + \

			all_loss = G_train_loss
			all_loss.backward()
			#G_train_loss.backward()
			G_optimizer.step()
			#embedding_loss.backward(retain_graph=True)

		print ("Epoch:{}\tIterations: {}/{}\t discriminator loss:{} \t generator loss:{} \t \t ".format(
																					 epoch,
																					 iteration + 1,
																					 num_batch,
		                                                                             D_train_loss.cpu().data.numpy()[0],
		                                                                             G_train_loss.cpu().data.numpy()[0],

				))
		train_dis_meter_loss.add(D_train_loss.cpu().data[0])
		train_gen_meter_loss.add(G_train_loss.cpu().data[0])
		if iteration == 0 and args.visdom:
			viz.images(
				255. * img_batch.cpu().data.numpy(),
				win=generated
			)

			viz.images(
				255. * g_res.cpu().data.numpy(),
				win=gt
			)
		if args.visdom and (iteration+1)%20==0:

			train_gen_loss_logger.log(total_iter, train_gen_meter_loss.value()[0])
			train_dis_loss_logger.log(total_iter, train_dis_meter_loss.value()[0])
			total_iter += 1
			train_dis_meter_loss.reset()
			train_gen_meter_loss.reset()
state = {
			'gen_state_dict': gen.state_dict(),
		}
filename = "params.pth"
torch.save(state, filename)