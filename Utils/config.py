import os
from yacs.config import CfgNode

def merge_cfg(cfg):
    cfg = CfgNode()
    cfg['mode'] = 'default'
    cfg['data_dir'] = 'default'
    cfg['save_dir'] = 'default'
    cfg['sub_dir'] = 'default'
    cfg['save_pred_dir'] = 'default'
    cfg['init_model'] = ''
    cfg['train_json'] = ''
    cfg['val_json'] = ''

    # model configs
    cfg['Net'] = ''
    cfg['pretrained'] = True
    cfg['drop_rate'] = 0.2
    cfg['resume'] = False

    # parameters
    cfg['batch_size'] = 512
    cfg['eval_batch_size'] = 512
    cfg['num_classes'] = 209
    cfg['epochs'] = 36
    cfg['print_per_epoch'] = 500
    cfg['eval_per_epoch'] = 2

    # optimize/schdule/loss
    cfg['loss'] = 'ce'
    cfg['lr_schdule'] = 'Step'
    cfg['lr'] = 5e-3
    cfg['weight_decay'] = 1e-6
    cfg['optim'] = 'adam'
    cfg['lr_decay_epochs'] = [20, 26, 32]
    cfg['class_weight'] = False # not use for loss weight

    # augmentation setting
    cfg['clahe'] = True
    cfg['clahe_prob'] = 0.2
    cfg['gray'] = True
    cfg['gray_prob'] = 0.01
    cfg['aug_proba'] = 0.5 # albumentations aug prob
    cfg['cut_size'] = 8 # cutout
    cfg['label_smooth'] = 0.01 # 0.0 for unuse
    cfg['mixup'] = False
    cfg['mixup_alpha'] = 1
    cfg['height'] = 224
    cfg['width'] = 224
    cfg['thread'] = 16

    cfg['weight_sampler'] = False # weight for data sampler
    cfg['weight_per_cls'] = None

    # NTS specify configs
    cfg['backbone'] = 'resnet' # should specify in NTS model
    cfg['PROPOSAL_NUM'] = 6
    cfg['CAT_NUM'] = 4

cfg = CfgNode()
merge_cfg(cfg)