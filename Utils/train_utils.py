import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


class cb_loss(nn.Module):
	"""cb_loss in cvpr2019"""
	def __init__(self, samples_per_cls, no_of_classes, loss_type='softmax', beta=0.9999, gamma=2.0):
		super(cb_loss, self).__init__()
		self.samples_per_cls = samples_per_cls
		self.no_of_classes = no_of_classes
		self.loss_type = loss_type
		self.beta = beta
		self.gamma = gamma
	
	def forward(self, logits, labels):
		"""
		labels: one-hot
		"""
		effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
		weights = (1.0 - self.beta) / np.array(effective_num)
		weights = weights / np.sum(weights) * self.no_of_classes # (nr_class, )权重归一化 * nr_class

		labels_one_hot = labels # .cpu()
		# labels_one_hot = F.one_hot(labels, self.no_of_classes).float() # (bs, nr_class)
		
		weights = torch.tensor(weights).float()
		weights = logits.new_tensor(weights)
		weights = weights.unsqueeze(0) # (1,nr_class)
		weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot # (bs, nr_class) 获得该样本的加权因子
		weights = weights.sum(1)
		weights = weights.unsqueeze(1) # (bs, 1)
		weights = weights.repeat(1, self.no_of_classes) # (bs, nr_class) 每个样本固定的一个weights，dim=1维度上的值都是相等的weight
		# from IPython import embed; embed()
		# labels_one_hot = logits.new_tensor(labels_one_hot)
		# weights = logits.new_tensor(weights)
		if self.loss_type == "focal":
			cb_loss = _focal_loss(labels_one_hot, logits, weights, self.gamma)
		elif self.loss_type == "sigmoid":
			cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
		elif self.loss_type == "softmax":
			pred = logits.softmax(dim=1)
			cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
		return cb_loss

	def _focal_loss(self, labels, logits, alpha, gamma):
		BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction = "none")
		if gamma == 0.0:
			modulator = 1.0
		else:
			modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))
		loss = modulator * BCLoss
		weighted_loss = alpha * loss
		focal_loss = torch.sum(weighted_loss)
		focal_loss /= torch.sum(labels)
		return focal_loss


class cross_entropy(nn.Module):
	""" Cross entropy that accepts soft targets"""

	def __init__(self, size_average=True, func_type='softmax'):
		super(cross_entropy, self).__init__()
		self.size_average = size_average
		self.func_type = func_type

	def forward(self, input, target):
		"""
		input: logits
		target: one_hot_label
		"""
		if self.func_type == 'softmax':
			logsoftmax = nn.LogSoftmax()
			if self.size_average:
				return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
			else:
				return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))
		else: # sigmoid
			return F.binary_cross_entropy_with_logits(input=input, target=target)


class focal_loss(nn.Module):
	def __init__(self, alpha=1., gamma=1.):
		super(focal_loss, self).__init__()
		self.alpha = alpha
		self.gamma = gamma

	def forward(self, inputs, targets, **kwargs):
		CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
		pt = torch.exp(-CE_loss)
		F_loss = self.alpha * ((1 - pt) ** self.gamma) * CE_loss
		return F_loss.mean()


class focal_loss_3d(nn.Module):
	def __init__(self, gamma=1.0, alpha=None, size_average=True):
		super(focal_loss_3d, self).__init__()
		self.gamma = gamma
		self.alpha = alpha
		if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
		if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
		self.size_average = size_average

	def forward(self, input, target):
		if input.dim() > 2:
			input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
			input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
			input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
		target = target.view(-1, 1)

		logpt = F.log_softmax(input)
		logpt = logpt.gather(1, target)
		logpt = logpt.view(-1)
		pt = Variable(logpt.data.exp())

		if self.alpha is not None:
			if self.alpha.type() != input.data.type():
				self.alpha = self.alpha.type_as(input.data)
			at = self.alpha.gather(0, target.data.view(-1))
			logpt = logpt * Variable(at)

		loss = -1 * (1 - pt) ** self.gamma * logpt
		if self.size_average:
			return loss.mean()
		else:
			return loss.sum()


def weights_init(m, init_f=nn.init.kaiming_normal_):
	gain = nn.init.calculate_gain('relu')
	if isinstance(m, nn.BatchNorm2d):
		m.weight.data.fill_(1)
		m.bias.data.zero_()
	elif isinstance(m, nn.Linear):
		init_f(m.weight.data, nonlinearity='relu')
		m.bias.data.zero_()


def adjust_learning_rate_cliff(optimizer, epoch, initial_lr):
	"""Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
	lr = initial_lr * (0.1 ** (epoch // 8))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def adjust_learning_rate_linear(optimizer, epoch, initial_lr):
	lr = initial_lr * (0.9 ** (epoch // 1))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def adjust_learning_rate_traingle(optimizer, epoch, max_lr, min_lr=0.0001, cycle=8):
	valid_epoch = epoch % cycle
	k = (max_lr - min_lr) / (cycle // 2)
	lr = max_lr - abs(valid_epoch - cycle // 2) * k
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def adjust_learning_rate_warmup(optimizer, epoch, max_lr, min_lr=0.0001, cycle=8):
	valid_epoch = epoch % cycle
	valid_max_lr = max_lr * (0.7 ** (epoch // cycle))
	delta_lr = (valid_max_lr - min_lr)
	k = delta_lr / cycle
	if epoch < cycle:
		if epoch <= cycle // 2:
			lr = min_lr + 0.5 * delta_lr + valid_epoch * k
		else:
			lr = max_lr - (epoch - cycle // 2) * 2 * k
	else:
		lr = valid_max_lr - (valid_epoch - cycle) * k
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def get_optimizer(cfg, model):
	param_groups = model.parameters()

	if cfg.TRAIN.OPTIM == 'Adam':
		optimizer = torch.optim.Adam(param_groups, lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
	else:
		optimizer = torch.optim.SGD(param_groups, lr=cfg.TRAIN.LR, momentum=0.9, nesterov=True, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

	return optimizer


def mixup_data(x, y, alpha=1.0, use_cuda=True):
	'''Returns mixed inputs, pairs of targets, and lambda'''
	if alpha > 0:
		lam = np.random.beta(alpha, alpha)
	else:
		lam = 1

	batch_size = x.size()[0]
	if use_cuda:
		index = torch.randperm(batch_size).cuda() # 返回0~batch_size-1的数组，不有序
	else:
		index = torch.randperm(batch_size)

	mixed_x = lam * x + (1 - lam) * x[index, :]
	y_a, y_b = y, y[index]
	return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
	return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class AverageMeter(object):
	"""Computes and stores average and current value for one epoch"""
	def __init__(self, name, fmt=':f'):
		self.name = name
		self.fmt = fmt
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

	def __str__(self):
		fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
		return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
	def __init__(self, num_batches, meters, prefix=""):
		self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
		self.meters = meters
		self.prefix = prefix

	def display(self, batch):
		entries = [self.prefix + self.batch_fmtstr.format(batch)]
		entries += [str(meter) for meter in self.meters]
		print('\t'.join(entries))

	def _get_batch_fmtstr(self, num_batches):
		num_digits = len(str(num_batches // 1))
		fmt = '{:' + str(num_digits) + 'd}'
		return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, labels, use_onehot):
	# FIXME: maybe do these as tensor would be faster
	preds = np.argmax(output.cpu().detach().numpy(), axis=1)
	if use_onehot:
		targets = np.argmax(labels.cpu().detach().numpy(), axis=1)
	else:
		targets = labels.cpu().detach().numpy()
	acc = metrics.accuracy_score(targets, preds)
	f1 = metrics.f1_score(targets, preds, average='macro')
	
	return acc, f1

def rm_model(params):
	ckpts = sorted(name for name in os.listdir(params['save_dir']) if name.startswith('model'))
	ckpts = ckpts[:-1] # keep model_best.pkl
	for i in ckpts:
		os.remove(os.path.join(params['save_dir'], i))

def set_logs(cfg):
	cfg.LOG_DIR = os.path.join(cfg.SAVE_DIR, "log/")

	if not os.path.exists(cfg.LOG_DIR):
		os.makedirs(cfg.LOG_DIR)
		# TODO: to be more elegant, use logging

def load_ckpt(save_dir):
	ckpts = glob(os.path.join(save_dir, '*.pkl'))
	if len(ckpts)>0:
		# ckpts.remove(os.path.join(save_dir, 'model_best.pkl'))
		ckpts = sorted(ckpts, key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
		cfg.INIT_MODEL = ckpts[-1] # 最新的
		print(cfg.INIT_MODEL)