# coding=utf-8
# NTS_net的脚本
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
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR

from DataSet.dataset import get_iwildcam_loader, data_prefetcher
from Utils.train_utils import cross_entropy, focal_loss, cb_loss, get_optimizer, set_logs, load_ckpt
from Utils.train_utils import AverageMeter, ProgressMeter, accuracy
from Utils.train_utils import mixup_data, mixup_criterion
from Utils.default import _C as cfg
from Utils.default import update_config
from Models import NTS

import click
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
	print("Use Device Number: {}".format(torch.cuda.device_count()))

def evaluate(model, data_loader, criterion, use_onehot=True):
	y_pred, y_true, losses=[],[],[]
	with torch.no_grad():
		inputs, labels, ids = data_loader.next()
		while inputs is not None:
			bs = inputs.size(0)
			if use_onehot:
				targets = np.argmax(labels.cpu().detach().numpy(), axis=1)
			else:
				targets = labels.cpu().detach().numpy()
			y_true.extend(targets)
			# output = model(inputs)
			_, concat_logits, _, _, _ = model(inputs)
			# --- calculate loss ---
			concat_loss = criterion(concat_logits, labels)
			# --- calculate accuarcy ---
			y_pred.extend(np.argmax(concat_logits.cpu().detach().numpy(), axis=1))
			losses.append(concat_loss.item())

			inputs, labels, ids = data_loader.next()

	acc = metrics.accuracy_score(y_true, y_pred)
	f1 = metrics.f1_score(y_true, y_pred, average='macro')
	loss_val = np.mean(losses)
	return loss_val, acc, f1

def main(cfg):
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

	model = NTS.attention_net(cfg, CAT_NUM=cfg.NET.CAT_NUM, topN=cfg.NET.PROPOSAL_NUM)
	print(model)

	# special for NTS
	raw_parameters = list(model.pretrained_model.parameters())
	part_parameters = list(model.proposal_net.parameters())
	concat_parameters = list(model.concat_net.parameters())
	partcls_parameters = list(model.partcls_net.parameters())
	
	raw_optimizer = torch.optim.SGD(raw_parameters, lr=cfg.TRAIN.LR, momentum=0.9, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
	concat_optimizer = torch.optim.SGD(concat_parameters, lr=cfg.TRAIN.LR, momentum=0.9, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
	part_optimizer = torch.optim.SGD(part_parameters, lr=cfg.TRAIN.LR, momentum=0.9, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
	partcls_optimizer = torch.optim.SGD(partcls_parameters, lr=cfg.TRAIN.LR, momentum=0.9, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

	param_num = sum([p.data.nelement() for p in model.parameters()])
	print("Number of model parameters: {} M".format(param_num / 1024 / 1024))

	model = model.cuda()
	model = DataParallel(model)
	model.train()

	train_data_loader, dev_data_loader = get_iwildcam_loader(params, mode=params['mode']) # train/eval的dataloader

	if params['lr_schedule'] == "Step":# True
		# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params['lr_decay_epochs'], gamma=0.2)
		schedulers = [MultiStepLR(raw_optimizer, milestones=cfg.TRAIN.LR_DECAY_EPOCHS, gamma=0.1),
                    MultiStepLR(concat_optimizer, milestones=cfg.TRAIN.LR_DECAY_EPOCHS, gamma=0.1),
                    MultiStepLR(part_optimizer, milestones=cfg.TRAIN.LR_DECAY_EPOCHS, gamma=0.1),
                    MultiStepLR(partcls_optimizer, milestones=cfg.TRAIN.LR_DECAY_EPOCHS, gamma=0.1)]
	elif params['lr_schedule'] == "Cosine":
		schedulers = [CosineAnnealingLR(raw_optimizer, T_max=(cfg.TRAIN.EPOCHS // 9) + 1, eta_min=1e-06),
                    CosineAnnealingLR(concat_optimizer, T_max=(cfg.TRAIN.EPOCHS // 9) + 1, eta_min=1e-06),
                    CosineAnnealingLR(part_optimizer, T_max=(cfg.TRAIN.EPOCHS // 9) + 1, eta_min=1e-06),
                    CosineAnnealingLR(partcls_optimizer, T_max=(cfg.TRAIN.EPOCHS // 9) + 1, eta_min=1e-06)
		]
	
	best_acc, best_f1, best_epoch, start_epoch = 0, 0, 0, 1
	# ------ Begin Resume -------
	if cfg.RESUME:
		load_ckpt(cfg.SAVE_DIR) # read history parameters from json
		ckpt = torch.load(cfg.INIT_MODEL, map_location="cuda") # already specify in load_params()
		print('=> Load checkpoint from ', cfg.INIT_MODEL)
		model.load_state_dict(ckpt['state_dict'])
		raw_optimizer.load_state_dict(ckpt['raw_optimizer'])
		part_optimizer.load_state_dict(ckpt['part_optimizer'])
		concat_optimizer.load_state_dict(ckpt['concat_optimizer'])
		partcls_optimizer.load_state_dict(ckpt['partcls_optimizer'])
		# optimizer.load_state_dict(ckpt['optimizer'])
		scheduler.load_state_dict(ckpt['schduler']) # FIXME: to check
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

	t0 = time()
	t1 = time()
	it = 0
	print('[INFO]Begin to train')
	use_onehot = cfg.LOSS.LOSS_TYPE != 'Focal'
	for epoch in range(start_epoch, cfg.TRAIN.EPOCHS + 1):
		print('=> Current Lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
		if cfg.TRAIN.LR_SCHEDULE:
			scheduler.step()

		train_loader = data_prefetcher(train_data_loader, label_type)
		inputs, labels, ids = train_loader.next() # ids没有用到
		i = 0
		batch_time = AverageMeter('Time', ':6.3f')
		data_time = AverageMeter('Data', ':6.3f')
		losses = AverageMeter('Loss', ':.4e')
		train_acc = AverageMeter('Acc', ':6.2f')
		train_f1 = AverageMeter('F1', ':6.2f')
		progress = ProgressMeter(
			len(train_data_loader),
			[batch_time, data_time, losses, train_acc, train_f1],
			prefix="Epoch: [{}]".format(epoch))

		while inputs is not None:
			bs = inputs.size(0)
			# mixup_now = np.random.random() < cfg.AUG.AUG_PROBA # 0.5 一半的概率mixup
			# if cfg.AUG.MIXUP and mixup_now: # True & 一半的概率
			# 	inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, cfg.AUG.MIXUP_ALPHA)

			raw_logits, concat_logits, part_logits, _, top_n_prob = model(inputs)

			# optimizer.zero_grad()
			raw_optimizer.zero_grad()
			part_optimizer.zero_grad()
			concat_optimizer.zero_grad()
			partcls_optimizer.zero_grad()

			raw_logits, concat_logits, part_logits, _, top_n_prob = model(inputs)
			if cfg.AUG.MIXUP and mixup_now:
				# TODO: to implement NTS with mixup
				# loss = mixup_criterion(criterion, output, labels_a, labels_b, lam) # mixup之后的图片也要根据mixup的obj算loss
				pass
			else:
				part_loss = NTS.list_loss(
					part_logits.view(bs * cfg.NET.PROPOSAL_NUM, -1),
					labels.max(axis=1)[1].unsqueeze(1).repeat(1, cfg.NET.PROPOSAL_NUM).view(-1)).view(bs, cfg.NET.PROPOSAL_NUM)
				raw_loss = criterion(raw_logits, labels)
				concat_loss = criterion(concat_logits, labels)
				rank_loss = NTS.ranking_loss(top_n_prob, part_loss, proposal_num=cfg.NET.PROPOSAL_NUM)

				CE = torch.nn.CrossEntropyLoss()
				partcls_loss = CE(
					part_logits.view(bs * cfg.NET.PROPOSAL_NUM, -1),
					labels.max(axis=1)[1].unsqueeze(1).repeat(1, cfg.NET.PROPOSAL_NUM).view(-1))
					# part_logits, (256,6,209) => (1536,209)
					# labels: (1536,)
				total_loss = raw_loss + rank_loss + concat_loss + partcls_loss

			total_loss.backward()

			raw_optimizer.step()
			part_optimizer.step()
			concat_optimizer.step()
			partcls_optimizer.step()

			if i % cfg.PRINT_STEP == 0:
				preds = np.argmax(concat_logits.cpu().detach().numpy(), axis=1) # argmax on logits
				if use_onehot:
					targets = np.argmax(labels.cpu().detach().numpy(), axis=1)
				else:
					targets = labels.cpu().detach().numpy()
				acc = metrics.accuracy_score(targets, preds)
				loss = concat_loss
				loss_val = loss.item()
				f1 = metrics.f1_score(targets,preds,average='macro')
				# train_log.append([epoch,i, loss_val, acc, f1])
				# print("epoch: %d, iter: %d, train_loss: %.4f, train_acc: %.4f, train_f1: %.4f, lr_rate: %.1e, time_cost_per_iter: %.4f s" % ( \
				# 	epoch, i, loss_val, acc, f1, (raw_optimizer.param_groups[0]['lr']), (time() - t1)/params['print_step']))
				tb_writer.add_scalar('train_loss', loss_val, it)
				# with open(params['log_dir'] + 'train.tsv', 'a') as f:
				# 	f.write('%05d\t%05d\t%f\t%f\t%f\n' % (epoch, i, loss_val, acc, f1))
				t1 = time()

			if (i+1) % params['eval_step'] == 0: # 95
				t2=time()
				model.eval()
				data_loader = data_prefetcher(dev_data_loader,label_type)
				loss_val, acc, f1 = evaluate(model, data_loader, criterion, use_onehot)
				model.train()
				dev_log.append([epoch, i, acc, f1])

				if f1 > best_f1:
					best_acc, best_f1, best_iter, best_epoch = acc, f1, i, epoch
				print('[Evaluation] -------------------------------')
				print("epoch: %d, test acc: %.4f, f1-score: %.4f, best-f1-score: %.4f, eval_time: %.4f s" % (
					epoch, acc, f1, best_f1,time()-t2))
				print('[Evaluation] -------------------------------')
				tb_writer.add_scalar('val_metrics/val_acc', acc, it)
				tb_writer.add_scalar('val_metrics/val_f1-score', f1, it)
				tb_writer.add_scalar('val_metrics/val_loss', loss_val, it)
				with open(params['log_dir'] + 'eval.tsv', 'a') as f:
					f.write('%05d\t%05d\t%f\t%f\n' % (epoch, i, acc, f1))
				
				save_model_path= os.path.join(params['save_dir'], 'model_%d_%d.pkl' % (epoch, i))
				# torch.save(model, save_model_path) # FIXME: this is bad for multi-gpu, use below instead
				torch.save({
					'state_dict': model.module.state_dict(),
					'schduler': scheduler.state_dict(),
					'raw_optimizer': raw_optimizer.state_dict(),
					'part_optimizer': part_optimizer.state_dict(),
					'concat_optimizer': concat_optimizer.state_dict(),
					'partcls_optimizer': partcls_optimizer.state_dict(),
					}, save_model_path)
				print('[INFO]save model to', save_model_path)

			inputs, labels, ids = train_loader.next()
			i += 1
			it += 1

	print("[INFO]Train is over, Time cost: %.1f hours..." % ((time()-t0) / 3600))
	# copy best_f1 model to model_best.pkl
	source = 'model_%d_%d.pkl' % (best_epoch, best_iter)
	source_path = os.path.join(params['save_dir'], source)
	target = 'model_best.pkl'
	target_path = os.path.join(params['save_dir'], target)
	try:
		shutil.copy(source_path, target_path)
		print("Save best model to {}: [epoch-iter: {:d}-{:d}/ f1-score: {:.4f}]".format(target_path, best_epoch, best_iter, best_f1))
	except IOError as e:
		print("Unable to copy file. %s" % e)
	except:
		print("Unexpected error:", sys.exc_info())

	# ---- Delete Useless ckpt
	ckpts = sorted(name for name in os.listdir(params['save_dir']) if name.startswith('model'))
	ckpts = ckpts[:-1]
	print("=> Start to clean checkpoint from {} to {}".format(ckpts[0], ckpts[-1]))
	for name in ckpts:
		os.remove(os.path.join(params['save_dir'], name))


def set_logs(cfg):
	cfg.LOG_DIR = os.path.join(cfg.SAVE_DIR, "log/")

	if not os.path.exists(cfg.LOG_DIR):
		os.makedirs(cfg.LOG_DIR)
		# TODO: to be more elegant, use logging

def load_ckpt(save_dir):
	# params_path = save_dir + 'log/parameters.json'
	# print('[INFO]Load params form', params_path)
	# params = json.load(fp=open(params_path, 'r'))
	ckpts = glob(os.path.join(save_dir, '*.pkl'))
	if len(ckpts)>0:
		# ckpts.remove(os.path.join(save_dir, 'model_best.pkl'))
		ckpts = sorted(ckpts, key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
		cfg.INIT_MODEL = ckpts[-1] # 最新的
		print(cfg.INIT_MODEL)
	# print(params)
	# return params

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