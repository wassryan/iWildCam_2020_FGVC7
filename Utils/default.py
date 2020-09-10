from yacs.config import CfgNode as CN

_C = CN()

# ---- BASIC SETTING ----
_C.MODE = ""
_C.SAVE_DIR = ""
_C.SAVE_PRED_DIR = ""
_C.INPUT_SIZE = (224, 224)
_C.RESUME = False
_C.INIT_MODEL = ""
_C.NUM_CLASSES = 209
_C.PRINT_STEP = 50
_C.CROSS_VALIDATION = False

# ---- DATASET BUILDER ----
_C.DATASET = CN()
_C.DATASET.DATA_DIR = ""
_C.DATASET.SUB_DIR = ""
_C.DATASET.TRAIN_JSON = ""
_C.DATASET.VAL_JSON = ""
_C.DATASET.TEST_JSON = ""

# ---- NETWORK BUILDER ----
_C.NET = CN()
_C.NET.TYPE = "tf_efficientnet_b0" # "NTS"
_C.NET.PRETRAINED = True
_C.NET.DROP_RATE = 0.2

# specify for NTS-NET
_C.NET.BACKBONE = "" # resnet
_C.NET.PROPOSAL_NUM = 6
_C.NET.CAT_NUM = 4

# ---- LOSS BUILDER ----
_C.LOSS = CN()
_C.LOSS.LOSS_TYPE = "CE" # ["CE", "Sigmoid_CE", "Focal", "CB_Loss"]
_C.LOSS.CLASS_WEIGHT = False
_C.LOSS.WEIGHT_PER_CLS = []
_C.LOSS.SAMPLES_PER_CLS = []
_C.LOSS.LOSS_WEIGHT = False

# ---- TRAIN BUILDER ----
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 256
_C.TRAIN.EPOCHS = 36
_C.TRAIN.EVAL_BATCH_SIZE = 256
_C.TRAIN.PRINT_PER_EPOCH = 500
_C.TRAIN.EVAL_PER_EPOCH = 2
_C.TRAIN.NUM_WORKER = 16
_C.TRAIN.WEIGHT_SAMPLER = False

# ---- SAMPLER/OPTIMIZER BUILDER ----
_C.TRAIN.LR_SCHEDULE = "Step" # "Cosine"
_C.TRAIN.LR = 5e-3
_C.TRAIN.LR_DECAY_EPOCHS = [20, 25, 32]
_C.TRAIN.WEIGHT_DECAY = 1e-6
_C.TRAIN.OPTIM = "Adam"

# ---- TEST BUILDER ----
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 64
_C.TEST.MODE = "" # ["infer", "infer_by_seq", "infer_by_seqv2"]

# ---- AUGMENTATION ----
_C.AUG = CN()
_C.AUG.CLAHE = True
_C.AUG.CLAHE_PROB = 0.2
_C.AUG.GRAY = True
_C.AUG.GRAY_PROB = 0.01
_C.AUG.AUG_PROBA = 0.5
_C.AUG.CUT_SIZE = 8
_C.AUG.LABEL_SMOOTH = 0.01
_C.AUG.MIXUP = False
_C.AUG.MIXUP_ALPHA = 1

def get_cfg_defaults():
  return _C.clone()

def update_config(cfg, args):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    # print(cfg)
    return cfg
