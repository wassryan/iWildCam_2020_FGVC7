import os
import json

def prepare_crossdata(K, M, ROOT, SAVE_ROOT, CROP_PATH, UNUSE):
	"""
	M: 0,1,2,..,K-1
	"""
	cat_id = 0
	t1 = time()
	dir_list = os.listdir(CROP_PATH)
	train = {}
	val = {}
	train_ann = []
	val_ann = []
	train_count = {}
	val_count = {}

	# 读取M=0时生成的category list
	if M!=0:
		with open(SAVE_ROOT + '/train_K0.json') as json_file:
			cat_list = json.load(json_file)
		cat_list = cat_list['categories']
		print("=> Read list from train_K0.json")
		
	for sub_id, sub_dir in enumerate(dir_list):
		if sub_dir in UNUSE: # 跳过6类unuse的类
			continue
		print("Starting Folder: ({}-{})".format(sub_id, sub_dir))
		path = os.path.join(CROP_PATH, sub_dir)
		imgs = glob(path + '/*')
		if len(imgs) < K:
			print("Error : Folder {} has less than {} imgs...".format(sub_dir, K))
			exit(0)
		else: # 正常划分
			for i, img in enumerate(imgs): # 按照K折分到train/val
				if i % K == M: # val
					if M == 0: # 用cat_id作为category_id
						train_ann, val_ann = gen_annotation(img, cat_id, sub_dir, 'val', train_ann, val_ann)
					else: # 参考M=0生成的category_id
						train_ann, val_ann = gen_annotation(img, cat_list[sub_dir]['category_id'], sub_dir, 'val', train_ann, val_ann)
					if M == 0: # M=0,则需要生成category对应；M=1,..参考M=0生成的对应列表
						train_count, val_count = gen_count(cat_id, sub_dir, 'val', train_count, val_count)
				else: # train
					if M == 0:
						train_ann, val_ann = gen_annotation(img, cat_id, sub_dir, 'train', train_ann, val_ann)
					else:
						train_ann, val_ann = gen_annotation(img, cat_list[sub_dir]['category_id'], sub_dir, 'train', train_ann, val_ann)
					if M == 0: # M=0,则需要生成category对应；M=1,..参考M=0生成的对应列表
						train_count, val_count = gen_count(cat_id, sub_dir, 'train', train_count, val_count)
		print("Done Folder: ({}-{})|| Cat_id: {}".format(sub_id, sub_dir, cat_id))
		cat_id += 1
	print("Total categories: {}".format(cat_id))
	print("All Folder Done... Total time-cost %.0f s" % (time() - t1))
	
	print("train_data has {} class, {} imgs".format(len(train_count.keys()), len(train_ann)))
	print("val_data has {} class, {} imgs".format(len(val_count.keys()), len(val_ann)))
	
	train['annotation'] = train_ann
	val['annotation'] = val_ann
	train['categories'] = train_count
	val['categories'] = val_count
	
	train_json = json.dumps(train)
	with open(os.path.join(SAVE_ROOT, 'train_K{}.json'.format(M)), 'w') as json_file:
		json_file.write(train_json)
	print("Save json file: 'train_K{}.json'".format(M))
	
	val_json = json.dumps(val)  
	with open(os.path.join(SAVE_ROOT, 'val_K{}.json'.format(M)), 'w') as json_file:
		json_file.write(val_json)
	print("Save json file: 'val_K{}.json'".format(M))

if __name__ == '__main__':
	ROOT = '/data/iwildcam-2020'
	SAVE_ROOT = ROOT + '/KCross'
	CROP_PATH = ROOT + '/animal_crops_224x224'
	UNUSE = ['unknown', 'unidentifiable', 'end', 'start', 'misfire', 'empty']
	K = 5 # 5折交叉验证
	if not os.path.exists(SAVE_ROOT):
		os.makedirs(SAVE_ROOT)

	for m in range(K):
		prepare_crossdata(K, m, ROOT, SAVE_ROOT, CROP_PATH, UNUSE)
		if m == 0:
			gen_cat_dict(SAVE_ROOT, m)
		print("---- {}/{} Finished ----".format(m, K))