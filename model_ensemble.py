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

def get_proba(x):
    proba = [eval(num) for num in x[1:-1].split()] # '[0.1, 0.2 ...]'
    return proba

def ensemble(params):
    print("[INFO] Start to ensemble models")
    print("[INFO] Process file: {}".format(params['target_file']))

    df = pd.DataFrame() # merge DataFrame

    for i, file in enumerate(params['target_file']):
        logits_path = params['save_pred_dir'] + file + '/log/model_best_logits.csv' 
        print("=> Process {}...".format(logits_path))
        csv_data = pd.read_csv(logits_path)
        temp = pd.DataFrame(csv_data)
        temp['Logits']=temp['Logits'].map(lambda x: get_proba(x)) # transform str into number
        temp = temp.rename(columns={'Logits': 'Logits' + str(i)})
        if len(df) == 0:
            df = temp
        else:
            df = pd.merge(df, temp, on=['Id'], how='inner')

    mean_probas = [list(df['Logits' + str(i)].values) for i in range(len(params['target_file']))]
    mean_probas = np.array(mean_probas) # (2, 26746, 209)
    mean_probas = np.mean(mean_probas, axis=0)
    
    df['Class'] = np.argmax(mean_probas, axis=1)
    df['Score'] = np.max(mean_probas, axis=1)
    print("Merge cols = ", df.columns)

    target_path = params['save_pred_dir'] + params['target_dir']
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    else:
        print("[INFO]This directory has already existed, Please remember to modify your 'target_dir'")
        if not click.confirm(
            "\033[1;31;40mContinue and override the former directory?\033[0m",
            default=False):
            exit(0)

    df[['Id', 'Class', 'Score']].to_csv(target_path + '/merged_preds.csv', index=False)
    print("[INFO] Merge done...")
    # {'Id': y_ids, 'Class': y_preds, 'Logits': list(logits_preds)}


def merge_result_by_clipv2(params):
    print("[INFO] ---- merge by clip strategyV2 ---- ")
    target_path = params['save_pred_dir'] + params['target_dir']
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
    print("[INFO]Load category correspondence from train_224.json")
    with open('/data/iwildcam-2020/train_224.json') as json_file:
        f2 = json.load(json_file)
    bg_class = f2['categories']['background']['category_id'] # get background class_id from train_aug.json
    data_file['Class'] = bg_class # default class_id as bg id
    data_file['Score'] = 0.0 # defaut score as 0.0
    
    for id, c, s in tqdm(zip(df_pred['Id'], df_pred['Class'], df_pred['Score']), total=len(df_pred['Id'])):
        data_file.loc[data_file['id'] == id, 'Class'] = c
        data_file.loc[data_file['id'] == id, 'Score'] = s
        
    data_file.to_csv(os.path.join(target_path, 'test_by_seq_tmp.csv'), index=False)
    # data_file = pd.read_csv(os.path.join(params['save_pred_dir'], 'test_by_seq_tmp.csv'))
    
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
                ids.extend(imgs)
                cls_ids = [most_class] * len(imgs)
                class_ids.extend(cls_ids)
                
    with open('/data/iwildcam-2020/category_224_list.json') as json_file: # 获取原category_id对应表
        class2cat = json.load(json_file)
    print("---- [Attention] background {} corresponde to {} ---- ".format(bg_class, class2cat[str(bg_class)]))
    o_category_ids = list(map(lambda x: class2cat[str(x)], class_ids)) # 映射回原图的class_id
    ids = list(map(lambda x: x.split('.')[0], ids)) # remove '.jpg'
    
    print("Submission Len: ", len(ids))
    sub_df = {'Id': ids, 'Category': o_category_ids}
    # 生成submission.csv
    save_path = os.path.join(target_path, params['target_dir'] + '_submission.csv')
    sub_df = pd.DataFrame(sub_df)

    print("=> Filter 24 imgs...")
    for over_id in over_id_list:
        sub_df.drop(sub_df.loc[sub_df['Id'] == over_id].index, inplace=True)
    print("Final submission data length: {}".format(sub_df.shape[0]))
    
    sub_df.to_csv(save_path, index=False)
    print("[INFO]Save submission.csv to {}".format(save_path))

def get_params():
	params = {
		'mode':'infer_by_seqv2',
		'data_dir': '/data/iwildcam-2020/', # 'data/bbox/cropped_image/', #data/bbox/cropped_image/'
		'save_pred_dir': '/data/iwildcam_output/final_output/',
        'target_file': ['NTS.data224', 'output_1.data224.adjust_lr'],
        'target_dir': 'ensemble1',
		'num_classes': 209,

	}
	print(params)
	return params

def main():
    params = get_params()
    
    # 将preds.csv中（original_image_id相同的小图依据策略合并结果）
    print("[INFO]Start to Ensemble...")
    t0 = time()
    ensemble(params)
    print("Start to Merge Result...")
    merge_result_by_clipv2(params)
    print('Merge Result Done, Time-cost %.0f min' % ((time() - t0) / 60))

if __name__ == '__main__':
	main()






