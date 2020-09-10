from __future__ import absolute_import

import os
import json
import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from sklearn.utils import shuffle

# Cutout: 在图像中生成num_holes个正方形的黑块区域
# HueSaturationValue: 随机色调、饱和度、值变化。
# RandomBrightnessContrast: 随机更改输入图像的亮度和对比度
# ShiftScaleRotate: 随机平移、缩放、旋转图片
def image_augment(p=.5, cut_size=8):
	imgaugment = A.Compose([
		A.HorizontalFlip(p=0.3),
		A.GaussNoise(p=.1),
		# A.OneOf([
		# A.Blur(blur_limit=3, p=.1),
		#	A.GaussNoise(p=.1),
		# ], p=0.2),
		A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT,
		                   value=(0, 0, 0), p=.3),
		A.RandomBrightnessContrast(p=0.3),
		A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.1),
		A.Cutout(num_holes=1, max_h_size=cut_size, max_w_size=cut_size, p=0.3)
	], p=p)

	return imgaugment

class iWildCam(Dataset):
	def __init__(self, cfg, mode='train'):
		self.mode = mode
		self.clahe = cfg.AUG.CLAHE # True
		self.gray = cfg.AUG.GRAY
		if 'train' in mode:
			clahe_prob = cfg.AUG.CLAHE_PROB # 0.2
			gray_prob = cfg.AUG.GRAY_PROB # 0.01
		elif mode in ['infer', 'infer_by_seq', 'infer_by_seqv2']: # infer时如果指定gray/clahe=True，则必用CLAHE/GRAY
			clahe_prob = 1
			gray_prob = 1
		else: # dev时
			clahe_prob =  int(cfg.AUG.CLAHE_PROB >= 1.0) # 0
			gray_prob = int(cfg.AUG.GRAY_PROB >= 1.0) # 0
		# augment, label_smooth
		if 'train' in mode:
			print('[INFO]Use train augmented mode')
			self.augment = cfg.AUG.AUG_PROBA > 0 # 0.5
			self.label_smooth = cfg.AUG.LABEL_SMOOTH # 0.01
		else: # dev/infer
			self.augment = False
			self.label_smooth = False
		self.one_hot = cfg.LOSS.LOSS_TYPE != 'Focal' if mode not in ['infer', 'infer_by_seq', 'infer_by_seqv2'] else False # 'loss'='ce' True
		self.num_classes = cfg.NUM_CLASSES
		self.root = cfg.DATASET.DATA_DIR #'/data/iwildcam-2020/'

		# mean_values = [0.3297, 0.3819, 0.3637] # vary for different dataset
		# std_values = [0.1816, 0.1887, 0.1877]

		mean_values = [0.3600, 0.3531, 0.3221] # RGB
		std_values = [1, 1, 1]

		self.resize = A.Resize(int(cfg.INPUT_SIZE[0] * 1.1), int(cfg.INPUT_SIZE[1] * 1.1), interpolation=cv2.INTER_CUBIC,
		                       p=1.0) # 64x64
		self.crop = A.RandomCrop(cfg.INPUT_SIZE[0], cfg.INPUT_SIZE[1], p=1.0) if 'train' in mode \
			else A.CenterCrop(cfg.INPUT_SIZE[0], cfg.INPUT_SIZE[1], p=1.0)

		if self.clahe: # infer: True(1), train: True(0.2), dev: True(0)
			self.imgclahe = A.CLAHE(clip_limit=2.0, tile_grid_size=(16, 16), p=clahe_prob)
		if self.gray: # infer: False, train: True, dev: True(0)
			self.imggray = A.ToGray(p=gray_prob)
		if self.augment: # infer: False
			self.imgaugment = image_augment(cfg.AUG.AUG_PROBA, cfg.AUG.CUT_SIZE)

		self.norm = transforms.Compose([
			transforms.ToTensor(),
			# transforms.Normalize(mean=mean_values,
			#                      std=std_values),
		]) # norm了 ToTensor() range [0, 255] -> [0.0,1.0]

		if mode == 'train_val': # train
			self.file_dir = self.root + cfg.DATASET.TRAIN_JSON # '/data/iwildcam-2020/' + 'train.json'
		elif mode == 'dev': # eval
			self.file_dir = self.root + cfg.DATASET.VAL_JSON # '/data/iwildcam-2020/' + 'val.json'
		elif mode == 'infer':
			self.file_dir = self.root + 'test.json' # '/data/iwildcam-2020/' + 'test.json'
		elif mode == 'infer_by_seq':
			self.file_dir = self.root + 'test_by_seq.json'
		elif mode == 'infer_by_seqv2':
			self.file_dir = self.root + cfg.DATASET.TEST_JSON
		else:
			print('does not exisit!', mode)

		print("=> Start Loading data from {}".format(self.file_dir))
		with open(self.file_dir) as json_file:
			f = json.load(json_file)

		if mode == 'train_val' or mode == 'dev':
			# list[dict], file_name(xx.jpg), id(xxx), category_id, category_name
			data_file = pd.DataFrame(f['annotation'])
			img_path = []
			for cname, fname in zip(data_file['category_name'], data_file['file_name']):
				img_path.append(os.path.join(self.root, cfg.DATASET.SUB_DIR, cname, fname))
		elif mode == 'infer':
			data_file = pd.DataFrame(f) # 'id', 'original_image_id'
			nr_data = len(data_file)
			print("Total data number: {}".format(nr_data)) # test.json: 73972
			# 过滤掉id为空的行，后续在merge_result()里直接对这些行，补类别为0
			data_file.drop(data_file.loc[data_file['id'] == ''].index, inplace=True)
			print("Filter no crops imgs {}, Remains imgs {}".format(nr_data - len(data_file), len(data_file)))# 33803, 40164
			img_path = []
			for fname, sub_dir in zip(data_file['id'], data_file['original_image_id']):
				img_path.append(os.path.join(self.root, 'test_crops', sub_dir, fname))
		elif mode == 'infer_by_seq':
			data_file = pd.DataFrame(f)# 'seq_id', 'location', 'clip_index', 'original_image_id', 'id'
			nr_data = len(data_file)
			print("Total data number: {}".format(nr_data)) # test_by_seq.json: 
			# 过滤掉id为空的行
			data_file.drop(data_file.loc[data_file['id'] == ''].index, inplace=True)
			print("Filter no crops imgs {}, Remains imgs {}".format(nr_data - len(data_file), len(data_file)))
			img_path = []
			for seq_id, loc, clip_idx, sub_dir, fname in zip(\
				data_file['seq_id'], data_file['location'], data_file['clip_index'], data_file['original_image_id'], data_file['id']):

				img_path.append(os.path.join(self.root, 'test_crops_seq_id', str(seq_id), str(loc), str(clip_idx), str(sub_dir), str(fname)))
		elif mode == 'infer_by_seqv2':
			data_file = pd.DataFrame(f) # 'seq_id', 'location', 'clip_index', 'id', 'file_exist'
			nr_data = len(data_file)
			print("Total data number: {}".format(nr_data)) # test_by_seqv2.json: 62894
			# 过滤掉file_exist为False的行
			data_file.drop(data_file.loc[data_file['file_exist'] == False].index, inplace=True)
			print("Filter no crops imgs {}, Remains imgs {}".format(nr_data - len(data_file), len(data_file)))# 36148, 26746
			img_path = []
			for seq_id, loc, clip_idx, fname in zip(\
				data_file['seq_id'], data_file['location'], data_file['clip_index'], data_file['id']):

				img_path.append(os.path.join(self.root, 'test_crops_seq_idv2', str(seq_id), str(loc), str(clip_idx), str(fname)))


		self.image_files = img_path
		self.image_ids = data_file['id'].values # 测试用，image_id: xxx_00000x.jpg; val时：xxx(xxx.jpg/xxx.png前缀)
		print('Dataset len:', len(self.image_files))
		if mode not in  ['infer', 'infer_by_seq', 'infer_by_seqv2']:
			self.labels = data_file['category_id'].values # 0~209


		# class sample weight
		if mode == 'train_val' and cfg.LOSS.CLASS_WEIGHT:
			self.image_files, self.labels = shuffle(self.image_files, self.labels, random_state=0) # shuffle
			class_sample_count = np.array([len(np.where(self.labels == t)[0]) \
				for t in np.unique(self.labels)])
			# second version: 0.671
			weight = class_sample_count.max() / class_sample_count
			# first version: 0.666
			# weight = 1. / class_sample_count
			cfg.LOSS.WEIGHT_PER_CLS = weight
			self.samples_weight = np.array([weight[t] for t in self.labels])

		if mode == 'train_val' and cfg.LOSS.LOSS_TYPE == 'cb_loss':
			# compute samples per class
			class_sample_count = [len(np.where(self.labels == t)[0]) \
				for t in np.unique(self.labels)]
			cfg.LOSS.SAMPLES_PER_CLS = class_sample_count

	def __getitem__(self, index):
		id = self.image_ids[index] # 测试用
		image = cv2.imread(self.image_files[index])
		image = self.resize(image=image)['image']

		if self.clahe: # infer时使用CLAHE
			image = self.imgclahe(image=image)['image']
		if self.augment: # F
			image = self.imgaugment(image=image)['image']
		if self.gray: # F
			image = self.imggray(image=image)['image']
		# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = self.crop(image=image)['image'] # random crop
		image = self.norm(image) # normalize

		if self.mode not in ['infer', 'infer_by_seq', 'infer_by_seqv2']:
			label = self.labels[index]
			if self.one_hot: # True
				label = np.eye(self.num_classes)[label]# 生成对应的one_hot
				if self.label_smooth > 0: # label_smooth = 0.01
					label = (1 - self.label_smooth) * label + self.label_smooth / self.num_classes
		else: # 如果mode=infer，生成全0的one-hot
			label = 0
			if self.one_hot:
				label = np.eye(self.num_classes)[label]

		return (image, label, id)

	def __len__(self):
		return len(self.image_files)

class data_prefetcher():
	def __init__(self, loader, label_type='float'):
		self.loader = iter(loader)
		self.stream = torch.cuda.Stream()
		self.label_type=label_type
		self.preload()

	def preload(self):
		try:
			self.next_input, self.next_target, self.next_ids = next(self.loader)
		except StopIteration:
			self.next_input = None
			self.next_target = None
			self.next_ids =None
			return
		with torch.cuda.stream(self.stream):
			self.next_input = self.next_input.cuda(non_blocking=True)
			self.next_target = self.next_target.cuda(non_blocking=True)
			#self.next_ids = self.next_ids.cuda(non_blocking=True)

			self.next_input = self.next_input.float()
			if self.label_type=='float':
				self.next_target = self.next_target.float()
			else:
				self.next_target = self.next_target.long()

	def next(self):
		torch.cuda.current_stream().wait_stream(self.stream)
		input = self.next_input
		target = self.next_target
		ids = self.next_ids
		self.preload()
		return input, target, ids


def get_iwildcam_loader(cfg, mode='train'):
	print("Mode: {}".format(mode))
	if mode == 'train' or mode == 'train_val' or mode == 'train_dev': # train_val
		train_data = iWildCam(cfg, mode=mode) # 定义一个取数据的 迭代器
		if cfg.TRAIN.WEIGHT_SAMPLER:
			train_sampler = WeightedRandomSampler(train_data.samples_weight, train_data.__len__()) 
			train_loader = torch.utils.data.DataLoader(
				train_data, batch_size=cfg.TRAIN.BATCH_SIZE,
				num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, pin_memory=True, sampler=train_sampler)
		else:
			train_loader = torch.utils.data.DataLoader(
				train_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
				num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, pin_memory=True)

		dev_data = iWildCam(cfg, mode='dev') # eval的数据

		dev_loader = torch.utils.data.DataLoader(
			dev_data, batch_size=cfg.TRAIN.EVAL_BATCH_SIZE, shuffle=False,
			num_workers=cfg.TRAIN.NUM_WORKER, drop_last=False, pin_memory=True)
		return train_loader, dev_loader
	elif mode in ['infer', 'infer_by_seq', 'infer_by_seqv2']:
		test_data = iWildCam(cfg, mode=mode)

		test_loader = torch.utils.data.DataLoader(
			test_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
			num_workers=cfg.TRAIN.NUM_WORKER, drop_last=False, pin_memory=True)
		return test_loader
	elif mode == 'val': # 仅用于验证模型的性能
		val_data = iWildCam(cfg, mode='dev') # eval的数据

		val_loader = torch.utils.data.DataLoader(
			val_data, batch_size=cfg.TRAIN.EVAL_BATCH_SIZE, shuffle=False,
			num_workers=cfg.TRAIN.NUM_WORKER, drop_last=False, pin_memory=True)
		return val_loader
	else:
		return None

if __name__ == '__main__':

	import cv2
	import argparse
	import os
	import sys
	sys.path.append("../")
	from Utils.default import update_config
	from Utils.default import _C as cfg
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', '-cfg', default='input yaml path')
	local_args = parser.parse_args()
	cfg = update_config(cfg, local_args)
	cfg.SAVE_DIR = os.path.join("/data/iwildcam_output", os.path.basename(local_args.config))
	# ! train_val
	train_data_loader, dev_data_loader = get_iwildcam_loader(cfg, mode='train_val') # 'train_val'
	train_loader = data_prefetcher(train_data_loader)
	# ! infer
	# test_data_loader = get_iwildcam_loader(params, mode='infer') # 'train_val'
	# test_loader = data_prefetcher(test_data_loader)	
	for i in range(1000):
		inputs, labels, ids = train_loader.next() # !train_val
		# inputs, labels, ids = test_loader.next() # !infer
		for j in range(inputs.shape[0]): # batch_size
			inp = inputs[j].cpu().numpy() * 255
			img = inp.transpose(1, 2, 0).astype(np.uint8)
			# label = labels[j].cpu().numpy().tolist()
			# print(img)
			# print(label)
			# print(img.shape, label.index(1))
			print(inputs.shape, labels.shape)
			# !imshow
			cv2.imshow("img", img)


