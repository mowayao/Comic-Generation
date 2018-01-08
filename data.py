from torch.utils.data import Dataset
import torch
import cv2
import numpy as np
import PIL.Image as pil_img
from torchvision.transforms import Compose, ToTensor, Resize


class data_folder(Dataset):
	def __init__(self, imgs, tags, hair_dim, eye_dim):
		super(data_folder, self).__init__()
		self.imgs = imgs
		self.tags = tags
		self.hair_dim = hair_dim
		self.eye_dim = eye_dim
		self.transforms = Compose([
			Resize(64),
			ToTensor()
		])
	def __getitem__(self, idx):
		img = self.imgs[idx]
		tag = self.tags[idx]
		img = pil_img.open(img)
		img = self.transforms(img)
		#hair_vec = torch.zeros(self.hair_dim)
		#hair_vec[tag[0]] = 1
		#eye_vec = torch.zeros(self.eye_dim)
		#eye_vec[tag[1]] = 1

		return img, tag[0], tag[1]
	def __len__(self):
		return len(self.imgs)