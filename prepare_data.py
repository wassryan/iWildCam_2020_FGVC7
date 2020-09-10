import os
import glob
import json
import pandas as pd
from time import time

ROOT = '/data/iwildcam-2020'
# CROP_PATH = ROOT + '/animal_crops'
CROP_PATH = ROOT + '/animal_crops_224x224'
train = {}
val = {}
train_ann = []
val_ann = []
train_count = {}
val_count = {}
# BG_CLASS = None

print("Generate json for {}...".format(CROP_PATH))
dir_list = os.listdir(CROP_PATH)
print("Sub folder num: ", len(dir_list))

unuse_class = ['unknown', 'unidentifiable', 'end', 'start', 'misfire', 'empty']

with open('/data/iwildcam-2020-fgvc7/iwildcam2020_train_annotations.json') as json_file:
    train_annotations_json = json.load(json_file)
df_cat = pd.DataFrame(train_annotations_json["categories"])

"""
根据目录生成train.json/ val.json
字段categories: dict{name: {count, orginal_category_id, category_id}}(每个类别有多少图片)
background的category_id=209，原category_id是0
字段annotation: list[dict], file_name(xx.jpg), id(xxx), category_id, category_name
"""

def gen_annotation(img_path, category_id, category_name, mode):
    tmp = {}
    img_name = os.path.basename(img_path)
    tmp["file_name"] = img_name
    tmp["id"] = img_name.split('.')[0]
    # if img_name.endswith("png"): # 对有一些奇怪命名的数据的特殊处理
    #     tmp["id"] = img_name.split('.')[0] + '.jpg'
    # else:
    #     tmp["id"] = img_name.split('.')[0]
    tmp["category_id"] = category_id
    tmp["category_name"] = category_name
    if mode == 'train':
        train_ann.append(tmp)
    else:
        val_ann.append(tmp)

# 按照train.json的categories字段生成{category_id: orginal_category_id}的对应
def gen_cat_dict():
    with open('/data/iwildcam-2020/train_224.json') as json_file: # 每个img下的crop小图个数
	    f = json.load(json_file)
    class2cat = {}
    for k, v in f['categories'].items():
        class2cat[v['category_id']] = v['orginal_category_id'] # key是str
    print("Total Category id: {}".format(len(class2cat.keys())))

    class2cat_json = json.dumps(class2cat)
    with open('/data/iwildcam-2020/category_224_list.json', 'w') as json_file:
        json_file.write(class2cat_json)
    print("background: {} to {}".format(\
        f['categories']['background']['category_id'], f['categories']['background']['orginal_category_id']))
    print("Save category_224_list.json...")

def gen_count(category_id, category_name, mode):
    if mode == 'train':
        # print(train_count.keys())
        if category_name == 'background':
            if category_name in train_count.keys(): # 若category_name已经在dict中
                train_count[category_name]['count'] += 1
            else:
                tmp = {}
                tmp['count'] = 1
                tmp['orginal_category_id'] = int(df_cat[df_cat['name'] == 'empty'].id.values[0])
                tmp['category_id'] = category_id
                train_count[category_name] = tmp
        else: # 为其他类
            if category_name in train_count.keys(): # 若category_name已经在dict中
                train_count[category_name]['count'] += 1
            else:
                tmp = {}
                tmp['count'] = 1
                tmp['orginal_category_id'] = int(df_cat[df_cat['name'] == category_name].id.values[0]) # np.int64转int
                tmp['category_id'] = category_id
                train_count[category_name] = tmp
    else:
        if category_name == 'background':
            if category_name in val_count.keys(): # 若category_name已经在dict中
                val_count[category_name]['count'] += 1
            else:
                tmp = {}
                tmp['count'] = 1
                tmp['orginal_category_id'] = int(df_cat[df_cat['name'] == 'empty'].id.values[0])
                tmp['category_id'] = category_id
                val_count[category_name] = tmp
        else: # 为其他类
            if category_name in val_count.keys(): # 若category_name已经在dict中
                val_count[category_name]['count'] += 1
            else:
                tmp = {}
                tmp['count'] = 1
                tmp['orginal_category_id'] = int(df_cat[df_cat['name'] == category_name].id.values[0])
                tmp['category_id'] = category_id
                val_count[category_name] = tmp

def split_train_val():
    # 按照类别分train/val, 1/9分
    # 若一个类别里只有一张图，给它分到train
    t1 = time()
    cat_id = 0
    for sub_id, sub_dir in enumerate(dir_list):
        # if sub_dir == 'background':
        #     BG_CLASS = sub_id
        if sub_dir in unuse_class: # 跳过6类unuse的类
            continue
        print("Starting Folder: ({}-{})".format(sub_id, sub_dir))
        path = os.path.join(CROP_PATH, sub_dir)
        imgs = glob.glob(path + '/*')
        if len(imgs) == 1: # 分到train
            print("-------[Warning] {} Only has one image --------".format(sub_dir))
            gen_annotation(imgs[0], cat_id, sub_dir, 'train')
            gen_count(cat_id, sub_dir, 'train')
        elif len(imgs) < 10: # 分最后1个img到val
            for i, img in enumerate(imgs[:-1]): # train
                gen_annotation(img, cat_id, sub_dir, 'train')
                gen_count(cat_id, sub_dir, 'train')
            gen_annotation(imgs[-1], cat_id, sub_dir, 'val')
            gen_count(cat_id, sub_dir, 'val')
        else: # 正常划分
            for i, img in enumerate(imgs): # 按照9/1分到train/val
                if i % 10 == 0: # val
                    gen_annotation(img, cat_id, sub_dir, 'val')
                    gen_count(cat_id, sub_dir, 'val')
                else: # train
                    gen_annotation(img, cat_id, sub_dir, 'train')
                    gen_count(cat_id, sub_dir, 'train')
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
    # print(train)
    train_json = json.dumps(train)
    with open(os.path.join(ROOT, 'train_224.json'), 'w') as json_file:
        json_file.write(train_json)
    print("Save json file: train_224.json")

    val_json = json.dumps(val)  
    with open(os.path.join(ROOT, 'val_224.json'), 'w') as json_file:
        json_file.write(val_json)
    print("Save json file: val_224.json")

def main():
    split_train_val()
    gen_cat_dict()

if __name__=='__main__':
	main()
	#get_test_size_multi_thread(thread_num=10)
	#merge_test_size_file()
