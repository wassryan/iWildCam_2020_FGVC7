# coding=utf-8
from __future__ import absolute_import, print_function
import argparse
import os
import json
import torch
import shutil
import time
import numpy as np
import pandas as pd
from sklearn import metrics
from glob import glob
from torch.nn import DataParallel
from torchsummary import summary
import torchvision.models as models

from DataSet.dataset import get_iwildcam_loader, data_prefetcher
from Utils.train_utils import cross_entropy, focal_loss, cb_loss, get_optimizer, set_logs, load_ckpt
from Utils.train_utils import AverageMeter, ProgressMeter, accuracy
from Utils.train_utils import mixup_data, mixup_criterion
from Utils.default import _C as cfg
from Utils.default import update_config
from Models.model_factory import create_model

import click
import warnings
warnings.filterwarnings("ignore")
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
	print("Use Device Number: {}".format(torch.cuda.device_count()))

def evaluate(model, data_loader, criterion, use_onehot=True):
	y_pred, y_true, losses=[],[],[]
	with torch.no_grad():
		inputs, labels, ids = data_loader.next()
		while inputs is not None:
			if use_onehot:
				targets = np.argmax(labels.cpu().detach().numpy(), axis=1)
			else:
				targets = labels.cpu().detach().numpy()
			y_true.extend(targets)
			output = model(inputs)
			# loss = criterion(output, labels)
			y_pred.extend(np.argmax(output.cpu().detach().numpy(), axis=1))
			# losses.append(loss.cpu().detach().numpy())

			inputs, labels, ids = data_loader.next()

	acc = metrics.accuracy_score(y_true, y_pred)
	f1 = metrics.f1_score(y_true, y_pred, average='macro')
	# loss_val=np.mean(losses)
	# return loss_val, acc, f1
	return acc, f1

def validate(val_loader, model, criterion, cfg, label_type, use_onehot):
	batch_time = AverageMeter('Time', ':6.3f')
	# losses = AverageMeter('Loss', ':.4e')
	val_acc = AverageMeter('Acc', ':6.2f')
	val_f1 = AverageMeter('F1', ':6.2f')
	# progress = ProgressMeter(
	# 	len(val_loader),
	# 	[batch_time, val_acc, val_f1],
	# 	prefix="Test: ")

	model.eval()
	data_loader = data_prefetcher(val_loader, label_type)
	i = 0
	inputs, labels, ids = data_loader.next()
	with torch.no_grad():
		end = time.time()
		while inputs is not None:
			output = model(inputs)
			acc, f1 = accuracy(output, labels, use_onehot)
			# losses.update(loss.item(), inputs.size(0))
			val_acc.update(acc, inputs.size(0))
			val_f1.update(f1, inputs.size(0))

			inputs, labels, ids = data_loader.next()
			i += 1

	return val_acc.avg, val_f1.avg

def train(train_data_loader, model, criterion, optimizer, epoch, cfg, label_type, use_onehot):
	batch_time = AverageMeter('Time', ':6.3f')
	data_time = AverageMeter('Data', ':6.3f')
	losses = AverageMeter('Loss', ':.4e')
	train_acc = AverageMeter('Acc', ':6.2f')
	train_f1 = AverageMeter('F1', ':6.2f')
	progress = ProgressMeter(
		len(train_data_loader),
		[batch_time, data_time, losses, train_acc, train_f1],
		prefix="Epoch: [{}]".format(epoch))

	model.train()
	train_loader = data_prefetcher(train_data_loader, label_type)
	i = 0
	end = time.time()
	inputs, labels, ids = train_loader.next() # ids没有用到
	while inputs is not None:
		data_time.update(time.time() - end)

		mixup_now = np.random.random() < cfg.AUG.AUG_PROBA # 0.5 一半的概率mixup
		if cfg.AUG.MIXUP and mixup_now: # True & 一半的概率
			inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, cfg.AUG.MIXUP_ALPHA)

		
		output = model(inputs)
		if cfg.AUG.MIXUP and mixup_now:
			loss = mixup_criterion(criterion, output, labels_a, labels_b, lam) # mixup之后的图片也要根据mixup的obj算loss
		else:
			if cfg.LOSS.CLASS_WEIGHT:
				# loss = CE(output, labels.max(axis=1)[1])
				loss = criterion(output, labels.max(axis=1)[1])
			else:
				loss = criterion(output, labels)

		# acc/f1 
		acc, f1 = accuracy(output, labels, use_onehot)
		losses.update(loss.item(), inputs.size(0))
		train_acc.update(acc, inputs.size(0))
		train_f1.update(f1, inputs.size(0))

		optimizer.zero_grad()			
		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % cfg.PRINT_STEP == 0:
			progress.display(i)

		inputs, labels, ids = train_loader.next()
		i += 1
	
	return losses.avg, train_acc.avg, train_f1.avg

def main(cfg, kcross=-1, K=-1):
	tensorboard_dir = os.path.join(cfg.SAVE_DIR, "tb_event")
	if not os.path.exists(cfg.SAVE_DIR):
		os.makedirs(cfg.SAVE_DIR)
	else:
		print("This directory has already existed, Please remember to modify your configs")
		if not click.confirm(
			"\033[1;31;40mContinue and override the former directory?\033[0m",
			default=False,
			):
			exit(0)
		if tensorboard_dir is not None and os.path.exists(tensorboard_dir):
			shutil.rmtree(tensorboard_dir)
	print("=> output model will be saved in {}".format(cfg.SAVE_DIR))
	tb_writer = SummaryWriter(tensorboard_dir)

	model = create_model(
		cfg.NET.TYPE,
		pretrained=cfg.NET.PRETRAINED,
		num_classes=cfg.NUM_CLASSES,
		drop_rate=cfg.NET.DROP_RATE,
		global_pool='avg',
		bn_tf=False,
		bn_momentum=0.99,
		bn_eps=1e-3,
		checkpoint_path=cfg.INIT_MODEL if cfg.INIT_MODEL != "" else None,
		in_chans=3)
	print(model)

	optimizer = get_optimizer(cfg, model)
	# use torchvision.models
 	# model = models.__dict__[params['Net']](num_classes=params['num_classes'])

	param_num = sum([p.data.nelement() for p in model.parameters()])
	print("=> Number of model parameters: {} M".format(param_num / 1024 / 1024))

	model = model.cuda()
	# summary(model, (3, cfg.INPUT_SIZE[0], cfg.INPUT_SIZE[1]))
	model = DataParallel(model)

	train_data_loader, dev_data_loader = get_iwildcam_loader(cfg, mode=cfg.MODE) # train/eval的dataloader

	if cfg.TRAIN.LR_SCHEDULE == 'Step':# True
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.LR_DECAY_EPOCHS, gamma=0.2)
	elif cfg.TRAIN.LR_SCHEDULE == 'Cosine':
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(cfg.TRAIN.EPOCHS // 9) + 1, eta_min=1e-06)
	else:
		raise NotImplementedError("Only Support lr_schdule(step, cosine)")

	best_acc, best_f1, best_epoch, start_epoch = 0, 0, 0, 1
	# ------ Begin Resume -------
	if cfg.RESUME:
		load_ckpt(cfg.SAVE_DIR) # read history parameters from json
		ckpt = torch.load(cfg.INIT_MODEL, map_location="cuda") # already specify in load_params()
		print('=> Load checkpoint from ', cfg.INIT_MODEL)
		model.load_state_dict(ckpt['state_dict'])
		optimizer.load_state_dict(ckpt['optimizer'])
		scheduler.load_state_dict(ckpt['scheduler'])
		start_epoch = ckpt['epoch'] + 1
		# best_acc = ckpt['best_acc']
		best_f1 = ckpt['best_f1']
		best_epoch = ckpt['best_epoch']

	if cfg.LOSS.LOSS_TYPE == 'CE':
		criterion = cross_entropy(func_type='softmax').to(device)
		if cfg.LOSS.WEIGHT_PER_CLS:
			CE = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(cfg.LOSS.WEIGHT_PER_CLS).float().to(device))
		label_type = 'float'
	elif cfg.LOSS.LOSS_TYPE == 'Sigmoid_CE':
		criterion = cross_entropy(func_type='sigmoid').to(device)
		label_type = 'float'		
	elif cfg.LOSS.LOSS_TYPE == 'Focal':
		criterion = focal_loss(gamma=1.0, alpha=1.0).to(device)
		label_type = 'long'
	elif cfg.LOSS.LOSS_TYPE == 'CB_loss': # FIXME: this is unsure implementation, low score
		criterion = cb_loss(cfg.LOSS.SAMPLES_PER_CLS, cfg.NUM_CLASSES, 'softmax').to(device)
		label_type = 'float'
	else:
		raise NotImplementedError("Not accessible loss type for: {}".format(cfg.LOSS.LOSS_TYPE))
	
	t0 = time.time()
	t1 = time.time()
	print('[INFO]Begin to train')
	use_onehot = cfg.LOSS.LOSS_TYPE != 'Focal'
	for epoch in range(start_epoch, cfg.TRAIN.EPOCHS + 1):
		print('=> Current Lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
		if cfg.TRAIN.LR_SCHEDULE:
			scheduler.step()

		if cfg.LOSS.CLASS_WEIGHT:
			train_loss, train_acc, train_f1 = \
				train(train_data_loader, model, CE, optimizer, epoch, cfg, label_type, use_onehot)
		else:
			train_loss, train_acc, train_f1 = \
				train(train_data_loader, model, criterion, optimizer, epoch, cfg, label_type, use_onehot)

		val_acc, val_f1 = validate(dev_data_loader, model, criterion, cfg, label_type, use_onehot)
		# TODO: this should also be done with the ProgressMeter
		print('=> [Epoch-{}] * Acc {:.3f} F1 {:.3f}'.format(epoch, val_acc, val_f1))

		is_best = val_f1 > best_f1
		best_f1 = max(val_f1, best_f1)
		best_epoch = epoch if is_best else best_epoch

		tb_writer.add_scalar('train_loss', train_loss, epoch)
		tb_writer.add_scalar('val_metrics/val_acc', val_acc, epoch)
		tb_writer.add_scalar('val_metrics/val_f1-score', val_f1, epoch)
				
		save_model_path= os.path.join(cfg.SAVE_DIR, 'model_{:03d}.pkl'.format(epoch))
		torch.save({
			'state_dict': model.state_dict(),
			'scheduler': scheduler.state_dict(),
			'optimizer': optimizer.state_dict(),
			'epoch': epoch,
			# 'best_acc': best_acc,
			'best_f1': best_f1,
			'best_epoch': best_epoch,
			}, save_model_path)
		print('=> save model to', save_model_path)

	print("=> Train is over, Time cost: {:.1f} hours...".format((time.time() - t0) / 3600))

	source = 'model_{:03d}.pkl'.format(best_epoch)
	source_path = os.path.join(cfg.SAVE_DIR, source)
	target = 'model_best.pkl'
	target_path = os.path.join(cfg.SAVE_DIR, target)
	try:
		shutil.copy(source_path, target_path)
		print("Save best model to {}: [Epoch: {:d} / f1-score: {:.4f}]".format(target_path, best_epoch, best_f1))
	except IOError as e:
		print("Unable to copy file. %s" % e)
	except:
		print("Unexpected error:", sys.exc_info())

	# ---- Delete Useless ckpt
	ckpts = sorted(name for name in os.listdir(cfg.SAVE_DIR) if name.startswith('model'))
	ckpts = ckpts[:-1]
	print("=> Start to clean checkpoint from {} to {}".format(ckpts[0], ckpts[-1]))
	for name in ckpts:
		os.remove(os.path.join(cfg.SAVE_DIR, name))

	if cfg.CROSS_VALIDATION:
		ksave_path = os.path.join(cfg.SAVE_DIR, 'kcross_model')
		if not os.path.exists(ksave_path):
			os.makedirs(ksave_path)
		kmodel_path = os.path.join(ksave_path, 'kcross_{}.pkl'.format(kcross))
		shutil.copy(target_path, kmodel_path)
		print("=> Save K-best model to {}...".format(kmodel_path))



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', '-cfg', default='input yaml path')
	local_args = parser.parse_args()
	cfg = update_config(cfg, local_args)
	cfg.SAVE_DIR = os.path.join("/data/iwildcam_output", os.path.basename(local_args.config))
	set_logs(cfg)
	print("=" * 40)
	print(cfg)
	print("=" * 40)
	if cfg.CROSS_VALIDATION:
		K = 5 # K-cross validation
		for i in range(K):
			# update params for K-cross
			cfg.DATASET.TRAIN_JSON = '/Kcross' + '/train_K{}.json'.format(i)
			cfg.DATASET.VAL_JSON = '/KCross' + '/val_K{}.json'.format(i)
			main(cfg, i, K)
			print("---- {}/{} Finished ---- ".format(i+1, K))
	else:
		main(cfg)