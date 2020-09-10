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
from Models import NTS
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

	model = torch.load(params['init_model'], map_location="cuda" if torch.cuda.is_available() else "cpu")
	print('Load model', params['init_model'])
	# model = model.to(device)
	model = model.cuda()
	model.eval()

	infer_loader = get_iwildcam_loader(params, mode='infer')
	infer_loader = data_prefetcher(infer_loader)
	y_preds, y_scores, y_ids = [], [], []
	logits_preds = []
	t1 = time()
	print('Begin to infer')
	with torch.no_grad():
		inputs, labels, ids = infer_loader.next()
		i = 0
		while inputs is not None: # 遍历batch_size上的多个图片
			_, concat_logits, _, _, _ = model(inputs) # vector
			output = torch.nn.functional.softmax(concat_logits, dim=-1)
			output = output.cpu().detach().numpy()
			logits_preds.extend(output) # logits-vector 也是softmax后的prob
			y_preds.extend(np.argmax(output, axis=1)) # list[class_id]
			y_scores.extend(np.max(output, axis=1))
			y_ids.extend(ids) # image_name: list[xxx_00000x.jpg]

			if (i+1) % params['print_step'] == 0:
				print("iter: %d,  time_cost_per_iter: %.4f s" % (i, (time() - t1)/params['print_step']))
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
	# read parameters from train scripts: parameteres.json
	with open(os.path.join(params['save_pred_dir'], 'parameters.json')) as file:
		train_params = json.load(file)
	params['backbone'] = train_params['backbone']
	params['CAT_NUM'] = train_params['CAT_NUM']
	params['PROPOSAL_NUM'] = train_params['PROPOSAL_NUM']

	ckpt = torch.load(params['init_model'])
	print('=> Load model', params['init_model'])
	model = NTS.attention_net(params, CAT_NUM=params['CAT_NUM'],topN=params['PROPOSAL_NUM'])
	model.load_state_dict(ckpt['state_dict'])
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
			_, concat_logits, _, _, _ = model(inputs)
			output = torch.nn.functional.softmax(concat_logits, dim=-1)
			output = output.cpu().detach().numpy()
			logits_preds.extend(output) # logits-vector 也是softmax后的prob
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
	save_path = params['save_pred_dir'] + params['init_model'].split('/')[-1].split('.')[0]+'_preds.csv'
	pred_df.to_csv(save_path, index=False)
	print("=> Save {} to {}".format(params['init_model'].split('/')[-1].split('.')[0]+'_preds.csv', save_path))

	logits_df = {'Id': y_ids, 'Class': y_preds, 'Logits': list(logits_preds)} # logits-vector, model embedding用
	logits_df = pd.DataFrame(logits_df)
	save_path = params['save_pred_dir']+params['init_model'].split('/')[-1].split('.')[0]+'_logits.csv'
	logits_df.to_csv(save_path, index=False)
	print("=> Save {} to {}".format(params['init_model'].split('/')[-1].split('.')[0]+'_logits.csv', save_path))
	print('pred done',pred_df.shape)

def merge_result_by_clip(params):

	print("[INFO] ---- merge by clip strategy ---- ")
	pred_path = params['save_pred_dir'] + params['init_model'].split('/')[-1].split('.')[0]+'_preds.csv'
	csv_data = pd.read_csv(pred_path)
	df_pred = pd.DataFrame(csv_data)
	print("Read preds result: {}".format(len(df_pred)))

	TEST_JSON = 'test_by_seq.json'
	with open(os.path.join(params["data_dir"], TEST_JSON)) as json_file:
		f = json.load(json_file)
	data_file = pd.DataFrame(f)# 'seq_id', 'location', 'clip_index', 'original_image_id', 'id'

	# add Class, Scores to test_by_seq.json DataFrame
	print("[INFO]Load category correspondence from train_224.json")
	with open('/data/iwildcam-2020/train_224.json') as json_file:
		f2 = json.load(json_file)
	bg_class = f2['categories']['background']['category_id'] # get background class_id from train_aug.json
	data_file['Class'] = bg_class # default class_id as bg id
	data_file['Score'] = 0.0 # defaut score as 0.0

	for id, o_id, c, s in tqdm(zip(df_pred['Id'], df_pred['O_Id'], df_pred['Class'], df_pred['Score']), total=len(df_pred['Id'])):
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
				

				# ! Only consider class counts number as metric(Attention to take non-bg as priority)
				if len(class_count) > 0:
					most_class = unique_class[class_count.argmax()]
				else:
					most_class = bg_class

				# set all images in this clip as 'most_class'
				imgs = np.unique(per_clip_df['original_image_id'].values)
				# set_trace()
				ids.extend(imgs)
				cls_ids = [most_class] * len(imgs)
				class_ids.extend(cls_ids)
	
	with open('/data/iwildcam-2020/category_224_list.json') as json_file: # 获取原category_id对应表
		class2cat = json.load(json_file)
	print("---- [Attention] background {} corresponde to {} ---- ".format(bg_class, class2cat[str(bg_class)]))
	o_category_ids = list(map(lambda x: class2cat[str(x)], class_ids)) # 映射回原图的class_id

	print()
	print("Submission Len: ", len(ids))
	
	sub_df = {'Id': ids, 'Category': o_category_ids}
	# 生成submission.csv
	save_path = params['save_pred_dir'] + params['init_model'].split('/')[-1].split('.')[0]+'_submission.csv'
	sub_df = pd.DataFrame(sub_df)

	# ! 修复bug，由于在resutlV2.json里面是添加了新的24个不需要的图片，这里需要把它们去除。。。
	for over_id in over_id_list:
		sub_df.drop(sub_df.loc[sub_df['Id'] == over_id].index, inplace=True)
	print("Final submission data length: {}".format(sub_df.shape[0]))

	sub_df.to_csv(save_path, index=False)
	print("[INFO]Save submission.csv to {}".format(save_path))
	
def merge_result_by_clipv2(params):

	print("[INFO] ---- merge by clip strategyV2 ---- ")
	pred_path = params['save_pred_dir'] + params['init_model'].split('/')[-1].split('.')[0]+'_preds.csv'
	csv_data = pd.read_csv(pred_path)
	df_pred = pd.DataFrame(csv_data)
	print("Read preds result: {}".format(len(df_pred)))

	TEST_JSON = 'test_by_seqv2.json'
	with open(os.path.join(params["data_dir"], TEST_JSON)) as json_file:
		f = json.load(json_file)
	data_file = pd.DataFrame(f)# 'seq_id', 'location', 'clip_index', 'id', 'file_exist'

	# add Class, Scores to test_by_seq.json DataFrame
	# get bg_class id
	print("[INFO]Load category correspondence from train_224.json")
	with open('/data/iwildcam-2020/train_224.json') as json_file:
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
	
	with open('/data/iwildcam-2020/category_224_list.json') as json_file: # 获取原category_id对应表
		class2cat = json.load(json_file)
	print("---- [Attention] background {} corresponde to {} ---- ".format(bg_class, class2cat[str(bg_class)]))
	o_category_ids = list(map(lambda x: class2cat[str(x)], class_ids)) # 映射回原图的class_id
	ids = list(map(lambda x: x.split('.')[0], ids)) # remove '.jpg'

	print()
	print("Submission Len: ", len(ids))

	sub_df = {'Id': ids, 'Category': o_category_ids}
	# 生成submission.csv
	save_path = params['save_pred_dir'] + params['init_model'].split('/')[-1].split('.')[0]+'_submission.csv'
	sub_df = pd.DataFrame(sub_df)

	print("=> Filter 24 imgs...")
	for over_id in over_id_list:
		sub_df.drop(sub_df.loc[sub_df['Id'] == over_id].index, inplace=True)
	print("Final submission data length: {}".format(sub_df.shape[0]))

	sub_df.to_csv(save_path, index=False)
	print("[INFO]Save submission.csv to {}".format(save_path))

def merge_result(params):
	"""
	从model_best_preds.csv中获取每个crop小图的分类类别(Class), 分数(Score)
	根据Id的前缀，获得original_image_id的list，o_image_id相同的Id一起判断
	组成一张图里的所有crop的list[class_id], list[score]
	再把class_id映射回原category_id
	"""
	pred_path = params['save_pred_dir'] + params['init_model'].split('/')[-1].split('.')[0]+'_preds.csv'
	csv_data = pd.read_csv(pred_path)
	df_pred = pd.DataFrame(csv_data)
	# df_pred = df_pred.sort_values(by=['Id'], ascending=True)

	print("Read preds result: {}".format(len(df_pred)))

	ids, class_ids = [], []
	O_Id = set(df_pred['O_Id'].values) # 原图id

	# image_id相同的放在一个for循环里判断，用O_Id索引吧
	for o_image_id in O_Id:
		# 属于同一个image
		df_per_image = df_pred[df_pred['O_Id'] == o_image_id]
		# print(o_image_id, len(df_per_image))
		class_list = set(df_per_image['Class'].values)
		# print("--- ", class_list)
		# set_trace()
		tmp_list = []
		for c in class_list: # 对于存在的每个类别
			df_c = df_per_image[df_per_image['Class'] == c]
			cnt = len(df_c)
			avg_score = float(df_c['Score'].sum()) / cnt
			tmp_list.append((c, cnt, avg_score))
		tmp_list.sort(key=lambda x: (x[1], x[2])) # 先按cnt排序，再按avg_score
		target = tmp_list[-1]
		target_c = target[0]
		target_score = target[2]
		ids.append(o_image_id)
		class_ids.append(target_c)
		"""
		1. 如果一只大象 三只羚羊，那我们就选羚羊做最后的predicted category
		2. n只大象n只羚羊，那就计算各自n只的prob，再比对，avg prob相差很多时候，选择大prob的类
		x 2. n只大象n只羚羊，那就计算各自n只的prob，再比对，avg prob相差不多，选择crop尺寸大的类【先不考虑，否则要在crop test的时候加新的字段】
		"""

	with open('/data/iwildcam-2020/test.json') as json_file:
		f = json.load(json_file)
	with open('/data/iwildcam-2020/train_224.json') as json_file:
		f2 = json.load(json_file)
	test_file = pd.DataFrame(f)
	empty_img = test_file[test_file['id'] == '']
	print("{} imgs need to define directly".format(len(empty_img)))
	empty_img_id = list(empty_img['original_image_id'].values)
	ids.extend(empty_img_id) # image_id
	
	bg_class = f2['categories']['background']['category_id'] # get background class_id from train_224.json
	bg_class_list = [bg_class for _ in range(len(empty_img_id))]
	class_ids.extend(bg_class_list)
	assert len(ids) == len(class_ids), "len(ids) doesn't match len(class_ids) {} v.s {}".format(len(ids), len(class_ids))

	with open('/data/iwildcam-2020/category_224_list.json') as json_file: # 获取原category_id对应表
		class2cat = json.load(json_file)
	print("---- [Attention] background {} corresponde to {} ---- ".format(bg_class, class2cat[str(bg_class)]))
	o_category_ids = list(map(lambda x: class2cat[str(x)], class_ids)) # 映射回原图的class_id
	# embed()

	sub_df = {'Id': ids, 'Category': o_category_ids}
	# 生成submission.csv
	save_path = params['save_pred_dir'] + params['init_model'].split('/')[-1].split('.')[0]+'_submission.csv'
	sub_df = pd.DataFrame(sub_df)

	# 由于在resutlV2.json里面是添加了新的24个不需要的图片，这里需要把它们去除
	for over_id in over_id_list:
		sub_df.drop(sub_df.loc[sub_df['Id'] == over_id].index, inplace=True)
	print("Final submission data length: {}".format(sub_df.shape[0]))

	sub_df.to_csv(save_path, index=False)
	print("Save submission.csv to {}".format(save_path))

	
def get_params():
	params = {
		'mode':'infer_by_seqv2',
		'data_dir': '/data/iwildcam-2020/',
		'save_pred_dir': '/data/iwildcam_output/final_output/NTS.resnet/log/',
		'init_model': '/data/iwildcam_output/final_output/NTS.resnet/model_best.pkl', # 'final_output/model_5_6827.pkl',

		'batch_size': 128,
		'num_classes': 209,
		'print_step':50,

		'clahe':True,
		'gray': False, # False, # 不使用Gray
		# 'white_balance': False,
		'height': 224, #128, # 64,
		'width': 224, #128, # 64,
		'threads': 16, # 2,
		'class_weight': False,
		'weight_sampler': False, # False, # use class_weight is bad

	}
	print(params)
	return params

# 模型预测embedding
def merge_pred_logit(pred_logit_files, save_dir):
	def get_proba(x):
		proba = [eval(num) for num in x[1:-1].split()]
		return proba

	df = pd.DataFrame()
	for ii, pf in enumerate(pred_logit_files):
		temp=pd.read_csv(pf)
		print('process file:',ii, pf, temp.shape)
		temp['Logits']=temp['Logits'].map(lambda x: get_proba(x))
		temp=temp.rename(columns={'Predicted':'Predicted'+str(ii),'Logits':'Logits'+str(ii)})
		if len(df)==0:
			df=temp
		else:
			df=pd.merge(df,temp,on=['Id'],how='inner')

	print('merged file',df.shape)
	mean_probas=[list(df['Logits'+str(ii)].values) for ii in range(len(pred_logit_files))]
	mean_probas=np.array(mean_probas)
	mean_probas=np.mean(mean_probas,axis=0)
	df['Probability'] = list(mean_probas)
	df['Predicted'] = np.argmax(mean_probas,axis=1)
	df[['Id','Predicted']].to_csv(save_dir + 'merged_submission.csv', index=False)
	df.to_csv(save_dir+'merged_proba.csv', index=False)
	print('mean proba done',df.shape)



def main():
	params = get_params()

	if params['mode'] == 'infer_by_seqv2':
		multi_inferv2(params)
	else:
		multi_infer(params) # 一张图中多个小图的同时infer，写到preds.csv中（每个小图一个id, original_image_id, category_id, score）

	# 将preds.csv中（original_image_id相同的小图依据策略合并结果）
	print("Start to Merge Result...")
	t0 = time()
	if params['mode'] == 'infer':
		merge_result(params)
	elif params['mode'] == 'infer_by_seq':
		merge_result_by_clip(params)
	elif params['mode'] == 'infer_by_seqv2':
		merge_result_by_clipv2(params)

	print('Merge Result Done, Time-cost %.0f min' % ((time() - t0) / 60))

def merge_pred():
	save_dir = 'final_output/pred/'
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	output_dir=[
		'output_0','output_1','output_2','output_3',
		'output_4','output_5','output_6','output_7',
		'output_8','output_9',
	]
	output_dir = ['final_output/' + x + '/log/' for x in output_dir]
	pred_logit_files =[]
	for outf in output_dir:
		logf = glob(outf+'*_logits.csv')
		pred_logit_files += logf
	print('pred_logit_files',len(pred_logit_files))
	print(pred_logit_files)
	merge_pred_logit(pred_logit_files, save_dir)

if __name__ == '__main__':
	main()






