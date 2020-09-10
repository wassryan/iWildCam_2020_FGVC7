# coding=utf-8
from __future__ import absolute_import, print_function
import os
import torch
from time import time
import pandas as pd
import numpy as np
from glob import glob
from DataSet.dataset import get_iwildcam_loader, data_prefetcher
from IPython import embed
from ipdb import set_trace
from tqdm import tqdm
import warnings
import json
import click
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)

over_id_list = [
	'89362ed4-21bc-11ea-a13a-137349068a90', '86994b3e-21bc-11ea-a13a-137349068a90', '8985bb98-21bc-11ea-a13a-137349068a90', 
	'8e940310-21bc-11ea-a13a-137349068a90', '8d705d8a-21bc-11ea-a13a-137349068a90', '88b99aae-21bc-11ea-a13a-137349068a90', 
	'9044a3b8-21bc-11ea-a13a-137349068a90', '8b91394e-21bc-11ea-a13a-137349068a90', '920ee4c4-21bc-11ea-a13a-137349068a90', 
	'9955d012-21bc-11ea-a13a-137349068a90', '8b8e02a6-21bc-11ea-a13a-137349068a90', '98da656c-21bc-11ea-a13a-137349068a90', 
	'8e930668-21bc-11ea-a13a-137349068a90', '89e09b26-21bc-11ea-a13a-137349068a90', '88a28616-21bc-11ea-a13a-137349068a90', 
	'9522d4fe-21bc-11ea-a13a-137349068a90', '950ed288-21bc-11ea-a13a-137349068a90', '882a533a-21bc-11ea-a13a-137349068a90', 
	'98552f5a-21bc-11ea-a13a-137349068a90', '8fff9dc2-21bc-11ea-a13a-137349068a90', '8a804608-21bc-11ea-a13a-137349068a90', 
	'8cc46b6a-21bc-11ea-a13a-137349068a90', '96bacf06-21bc-11ea-a13a-137349068a90', '8ea6a768-21bc-11ea-a13a-137349068a90'
]

def multi_infer(params):

	model = torch.load(cfg.INIT_MODEL, map_location="cuda" if torch.cuda.is_available() else "cpu")
	print('Load model', cfg.INIT_MODEL)
	# model = model.to(device)
	model = model.cuda()
	model.eval()

	infer_loader = get_iwildcam_loader(cfg, mode='infer')
	infer_loader = data_prefetcher(infer_loader)
	y_preds, y_scores, y_ids = [], [], []
	logits_preds = []
	t1 = time()
	print('Begin to infer')
	with torch.no_grad():
		inputs, labels, ids = infer_loader.next()
		i = 0
		while inputs is not None: # 遍历batch_size上的多个图片
			output = model(inputs) # vector
			output = torch.nn.functional.softmax(output, dim=-1)
			output = output.cpu().detach().numpy()
			logits_preds.extend(output) # logits-vector 也是softmax后的prob
			y_preds.extend(np.argmax(output, axis=1)) # list[class_id]
			y_scores.extend(np.max(output, axis=1))
			y_ids.extend(ids) # image_name: list[xxx_00000x.jpg]

			if (i+1) % 40 == 0:
				print("iter: %d,  time_cost_per_iter: %.4f s" % (i, (time() - t1) / 40]))
				t1 = time()
			i += 1
			inputs, labels, ids = infer_loader.next()

	O_ids = list(map(lambda x: x.split('_')[0], y_ids))
	# 'Id': [xxx_000.jpg, yyy_000.jpg],'O_Id': [xxx, yyy], 'Class': [class_id], 'Score': [0.1]
	pred_df = {'Id': y_ids, 'O_Id': O_ids, 'Class': y_preds, 'Score': y_scores}
	pred_df = pd.DataFrame(pred_df)
	save_path = params['save_pred_dir'] + params['init_model'].split('/')[-1].split('.')[0]+'_preds.csv'
	pred_df.to_csv(save_path, index=False)
	print("Save {} to {}".format(params['init_model'].split('/')[-1].split('.')[0]+'_preds.csv', save_path))

	logits_df = {'Id': y_ids, 'Class': y_preds, 'Logits': list(logits_preds)} # logits-vector, model embedding用
	logits_df = pd.DataFrame(logits_df)
	save_path = params['save_pred_dir']+params['init_model'].split('/')[-1].split('.')[0]+'_logits.csv'
	logits_df.to_csv(save_path, index=False)
	print("Save {} to {}".format(params['init_model'].split('/')[-1].split('.')[0]+'_logits.csv', save_path))
	print('pred done',pred_df.shape)

def multi_inferv2(params):

	model = torch.load(params['init_model'])
	print('=> Load model', params['init_model'])
	model = model.cuda()
	model.eval()

	infer_loader = get_iwildcam_loader(params, mode=params['mode'])
	infer_loader = data_prefetcher(infer_loader)
	y_preds, y_scores, y_ids = [], [], []
	logits_preds = []
	t1 = time()
	print('=> Begin to infer')
	with torch.no_grad():
		inputs, labels, ids = infer_loader.next()
		i = 0
		while inputs is not None: # 遍历batch_size上的多个图片
			output = model(inputs) # vector
			output = torch.nn.functional.softmax(output, dim=-1)
			output = output.cpu().detach().numpy()
			logits_preds.extend(output)
			y_preds.extend(np.argmax(output, axis=1)) # list[class_id]
			y_scores.extend(np.max(output, axis=1))
			y_ids.extend(ids) # image_name: list[xxx_00000x.jpg]

			if (i+1) % params['print_step'] == 0:
				print("iter: %d,  time_cost_per_iter: %.4f s" % (i, (time() - t1)/params['print_step']))
				t1 = time()
			i += 1
			inputs, labels, ids = infer_loader.next()

	O_ids = list(map(lambda x: x.split('_')[0], y_ids))
	# 'Id': [xxx.jpg, yyy.jpg], 'Class': [class_id], 'Score': [0.1]
	print("=> Pred Data Len: {}".format(len(y_ids)))
	pred_df = {'Id': y_ids, 'Class': y_preds, 'Score': y_scores}
	pred_df = pd.DataFrame(pred_df)
	save_path = os.path.join(params['save_pred_dir'], params['init_model'].split('/')[-1].split('.')[0]+'_preds.csv')
	pred_df.to_csv(save_path, index=False)
	print("=> Save {} to {}".format(params['init_model'].split('/')[-1].split('.')[0]+'_preds.csv', save_path))

	logits_df = {'Id': y_ids, 'Class': y_preds, 'Logits': list(logits_preds)} # logits-vector, model embedding用
	logits_df = pd.DataFrame(logits_df)
	save_path = os.path.join(params['save_pred_dir'], params['init_model'].split('/')[-1].split('.')[0]+'_logits.csv')
	logits_df.to_csv(save_path, index=False)
	print("=> Save {} to {}".format(params['init_model'].split('/')[-1].split('.')[0]+'_logits.csv', save_path))
	# print('pred done',pred_df.shape)

	
# this version is much better than merge_result_by_clip() 0.819 vs. 0.780
def merge_result_by_clipv2(params):

	print("[INFO] ---- merge by clip strategyV2 ---- ")
	target_path = params['target_dir']
	pred_path = target_path + '/merged_preds.csv'
	csv_data = pd.read_csv(pred_path)
	df_pred = pd.DataFrame(csv_data)
	print("Read preds result: {}".format(len(df_pred)))

	TEST_JSON = 'test_by_seqv2.json'
	with open(os.path.join(params["data_dir"], TEST_JSON)) as json_file:
		f = json.load(json_file)
	data_file = pd.DataFrame(f)# 'seq_id', 'location', 'clip_index', 'id', 'file_exist'

	# add Class, Scores to test_by_seq.json DataFrame
	# get bg_class id
	print("[INFO]Load category correspondence from train_K0.json")
	with open('/data/iwildcam-2020/KCross/train_K0.json') as json_file:
		f2 = json.load(json_file)
	bg_class = f2['categories']['background']['category_id'] # get background class_id from train_aug.json
	data_file['Class'] = bg_class # default class_id as bg id
	data_file['Score'] = 0.0 # defaut score as 0.0

	for id, c, s in tqdm(zip(df_pred['Id'], df_pred['Class'], df_pred['Score']), total=len(df_pred['Id'])):
		data_file.loc[data_file['id'] == id, 'Class'] = c
		data_file.loc[data_file['id'] == id, 'Score'] = s
	
	data_file.to_csv(os.path.join(params['save_pred_dir'], 'test_by_seq_tmp.csv'), index=False)

	# data_file = pd.read_csv(os.path.join(params['save_pred_dir'], 'test_by_seq_tmp.csv'))
	
	print("[INFO]Start to Merge Result")
	# Clip Strategy
	unique_seq_id = data_file['seq_id'].unique()
	print("Unique Seq id: {}".format(len(unique_seq_id)))

	ids, class_ids = [], []
	for seq_id in tqdm(unique_seq_id):
		loc_df = data_file[data_file['seq_id'] == seq_id]
		locations = loc_df['location'].unique()

		for loc in locations:
			clip_df = loc_df[loc_df['location'] == loc]
			clips = clip_df['clip_index'].unique()

			for clip_index in clips:
				per_clip_df = clip_df[clip_df['clip_index'] == clip_index]
				preds_class, preds_score = per_clip_df['Class'], per_clip_df['Score']
				# remove all bg_class, get max among non-bg class
				preds_class = np.array(preds_class)
				preds_class = np.delete(preds_class, np.where(preds_class == bg_class)[0])
				unique_class, class_count = np.unique(preds_class, return_counts=True)
				# most_class = unique_class[class_count.argmax()]

				# Only consider class counts number as metric(Attention to take non-bg as priority)
				if len(class_count) > 0:
					most_class = unique_class[class_count.argmax()]
				else:
					most_class = bg_class

				# set all images in this clip as 'most_class'
				imgs = np.unique(per_clip_df['id'].values)
				# set_trace()
				ids.extend(imgs)
				cls_ids = [most_class] * len(imgs)
				class_ids.extend(cls_ids)
	
	with open('/data/iwildcam-2020/KCross/category_Kcross_list.json') as json_file: # 获取原category_id对应表
		class2cat = json.load(json_file)
	print("---- [Attention] background {} corresponde to {} ---- ".format(bg_class, class2cat[str(bg_class)]))
	o_category_ids = list(map(lambda x: class2cat[str(x)], class_ids)) # 映射回原图的class_id
	ids = list(map(lambda x: x.split('.')[0], ids)) # remove '.jpg'

	print()
	print("Submission Len: ", len(ids))

	sub_df = {'Id': ids, 'Category': o_category_ids}
	# 生成submission.csv
	save_path = os.path.join(target_path, 'cross_submission.csv')
	sub_df = pd.DataFrame(sub_df)

	print("=> Filter 24 imgs...")
	for over_id in over_id_list:
		sub_df.drop(sub_df.loc[sub_df['Id'] == over_id].index, inplace=True)
	print("Final submission data length: {}".format(sub_df.shape[0]))

	sub_df.to_csv(save_path, index=False)
	print("=> Save submission.csv to {}".format(save_path))

def get_proba(x):
    proba = [eval(num) for num in x[1:-1].split()] # '[0.1, 0.2 ...]'
    return proba

def ensemble(params):
    print("=> Start to ensemble models")
    print("=> Process file: {}".format(params['target_file']))

    df = pd.DataFrame() # merge DataFrame

    logits_path = os.listdir(params['save_pred_dir'])
    logits_path = list(filter(lambda p: p.endswith('logits.csv'), logits_path))
    print("=> Logits file: ", logits_path)
    for i, p in enumerate(logits_path):
        logits_path = os.path.join(params['save_pred_dir'], p)
        print("=> Process {}...".format(logits_path))
        csv_data = pd.read_csv(logits_path)
        temp = pd.DataFrame(csv_data)
        temp['Logits']=temp['Logits'].map(lambda x: get_proba(x)) # transform str into number
        # set_trace()
        temp = temp.rename(columns={'Logits': 'Logits' + str(i)})
        if len(df) == 0:
            df = temp
        else:
            df = pd.merge(df, temp, on=['Id'], how='inner')

    mean_probas = [list(df['Logits' + str(i)].values) for i in range(len(params['target_file']))]
    # set_trace()
    mean_probas = np.array(mean_probas) # (2, 26746, 209)
    mean_probas = np.mean(mean_probas, axis=0)
    
    df['Class'] = np.argmax(mean_probas, axis=1)
    df['Score'] = np.max(mean_probas, axis=1)
    print("Merge cols = ", df.columns)

    target_path = params['target_dir']
    # if not os.path.exists(target_path):
    #     os.makedirs(target_path)
    # else:
    #     print("[INFO]{} has already existed, Please remember to modify your 'target_dir'".format(target_path))
    #     if not click.confirm(
    #         "\033[1;31;40mContinue and override the former directory?\033[0m",
    #         default=False):
    #         exit(0)

    df[['Id', 'Class', 'Score']].to_csv(target_path + '/merged_preds.csv', index=False)
    print("=> Merge done...")

def get_params():
	params = {
		'mode':'infer_by_seqv2',
		'data_dir': '/data/iwildcam-2020/', # 'data/bbox/cropped_image/', #data/bbox/cropped_image/'
		# 'save_pred_dir': '/data/iwildcam_output/final_output/output_1.data224.efficientnet.cosine/log/',
		'init_model': '/data/iwildcam_output/final_output/output_1.data224.efficientnet.cosine/', # 'final_output/model_5_6827.pkl',
		'batch_size': 512,
		'num_classes': 209,
		'print_step': 10,

		'clahe':True,
		'gray': False, # 不使用Gray
		# 'white_balance': False,
		'height': 224, #128, # 64,
		'width': 224, #128, # 64,
		'threads': 16, # 2,
		'class_weight': False,
		'weight_sampler': False, # False, # use class_weight is bad
	}
	print(params)
	return params

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', '-cfg', default='input yaml path')
	local_args = parser.parse_args()
	cfg = update_config(cfg, local_args)
    t0 = time()
    # params = get_params()
	SAVE_ROOT = "/data/iwildcam_output"
    root = os.path.join(SAVE_ROOT, os.path.basename(local_args.config))
    kroot = os.path.join(root, 'kcross_model')
    logits_path = os.path.join(root, 'kcross_csv')
    if not os.path.exists(logits_path):
        os.makedirs(logits_path)
    params['save_pred_dir'] = logits_path
    print("=> logits file will save in {}".format(params['save_pred_dir']))
    if cfg.TEST.MODE == 'infer_by_seqv2':
        nr_model = os.listdir(kroot)
        for i, m in enumerate(nr_model):
            # --- Redirect save path ---
            cfg.INIT_MODEL = os.path.join(kroot, m)
            print("[{}/{}] Start to Infer {}".format(i+1, len(nr_model), m))
            multi_inferv2(params)
    else:
        multi_infer(params) # 一张图中多个小图的同时infer，写到preds.csv中（每个小图一个id, original_image_id, category_id, score）

    # ensemble likes
    pkl_file = [pkl for pkl in os.listdir(kroot) if pkl.endswith('.pkl')]
    params['target_file'] = pkl_file # '*.pkl'
    params['target_dir'] = root # '/data/.../'
    ensemble(params)

    print("Start to Merge Result...")
    merge_result_by_clipv2(params)
    print('Merge Result Done, Time-cost %.0f min' % ((time() - t0) / 60))

if __name__ == '__main__':
	main()






