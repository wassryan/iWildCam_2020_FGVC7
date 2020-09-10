import os
import json
import cv2
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt
# from PIL import Image, ImageFile
from collections import defaultdict
from multiprocessing import Pool, Manager


DATA_ROOT = '/data/iwildcam-2020-fgvc7'
SAVE_ROOT = '/data/iwildcam-2020-fgvc7_crop'
TEST_ROOT = '/data/iwildcam-2020'
TEST_SAVE_ROOT = TEST_ROOT + '/test_crops'
DATA_SET = 'train'
MAX_COUNT = 6
NUM_PROCESS = 4

if not os.path.exists(SAVE_ROOT):
    os.makedirs(SAVE_ROOT)

# 本脚本既
# 1. 做按照detector_result.json的detection crop小图（卡score阈值）
# 2. 按照类别folder存放小图
# 3. 统计train.json里面应该有的类别，但是detector_result.json没有出现的类别
# 4. 统计train.json里面对该图有类别判定，但是detector对该图检测不到动物：miss={'category_id': cnt}, 
# 5. 统计detector.json框出的小图的类别：obj={'category_id': cnt}
# 应该把annotations里面的count也放到csv中，辅助来判断，有多少个动物，然后对box排序，取count个动物

"""
df_train = pd.DataFrame({'id': [item['id'] for item in train_data['annotations']],
                        'category_id': [item['category_id'] for item in train_data['annotations']],
                        'image_id': [item['image_id'] for item in train_data['annotations']],
                        'file_name': [item['file_name'] for item in train_data['images']]})
"""
# !Train_Data
csv_data = pd.read_csv('/data/iwildcam-2020-fgvc7/filter_train_img.csv')
df_train = pd.DataFrame(csv_data)

# # select a small batch for debug
# df_train = df_train[:30]

num_items = len(df_train)
length = math.ceil(num_items / NUM_PROCESS)
print("length of train data: {}".format(num_items))

img_names = df_train['file_name'].values
img_id = df_train['image_id'].values
cat_id = df_train['category_id'].values # 若cat_id为0，表明是empty类别

# 只读训练数据categories列表，用于id和category对应
with open('/data/iwildcam-2020-fgvc7/iwildcam2020_train_annotations.json') as json_file:
    train_annotations_json = json.load(json_file)
df_cat = pd.DataFrame(train_annotations_json["categories"])

# 用megadetector.json
with open('/data/iwildcam-2020-fgvc7/iwildcam2020_megadetector_resultsV2.json', encoding='utf-8') as json_file:
    megadetector_results = json.load(json_file)

megadetector_results_df = pd.DataFrame(megadetector_results["images"])

# test
with open('/data/iwildcam-2020-fgvc7/iwildcam2020_test_information.json') as json_file:
    test_json = json.load(json_file)
df_test = pd.DataFrame(test_json['images'])

test_items = len(df_test)
test_length = math.ceil(test_items / NUM_PROCESS)
print("Test Data Number: {}".format(test_items))

test_img_names = df_test['file_name'].values
test_img_id = df_test['id'].values

m = Manager()
share_cat_count = m.dict()
share_miss_cat_count = m.dict()
share_class_count = m.dict()
share_test_count = m.list() # 小图id和原图id的对应表
share_crop_count = m.dict() # 每个id下有几张图片
lock = m.Lock()

def CropTrainImg(i, cat_count, miss_cat_count, class_count, lock):
    t0 = time()
    start = length * i
    end = min(length * (i + 1), num_items)
    print("start process - {}, deal [{}, {}] data".format(i, start, end))
    cur_img_names = img_names[start: end]
    cur_img_id = img_id[start: end]
    cur_cat_id = cat_id[start: end]
    for im_name, id, cat in tqdm(zip(cur_img_names, cur_img_id, cur_cat_id), total=len(cur_img_names)):

        if int(cat) == 0: # !跳过empty的图片
            continue
        cat_name = df_cat[df_cat.id == int(cat)].name.values[0]
        im_path = os.path.join(DATA_ROOT, DATA_SET, im_name)
        img = cv2.imread(im_path)
        detections = megadetector_results_df[megadetector_results_df.id == id].detections.values[0]

        if len(detections) == 0: # !train.json的标签有类别，但是detector的结果是没检测有box
            lock.acquire()
            if cat_name not in miss_cat_count.keys():
                miss_cat_count[cat_name] = 1
            else:
                miss_cat_count[cat_name] += 1
            lock.release()

            # 保存miss的图片, 目录名为为类别名
            c_name = cat_name.replace(' ', '_')
            miss_path = os.path.join(SAVE_ROOT, 'miss_crop', c_name)
            if not os.path.exists(miss_path):
                os.makedirs(miss_path)
            cv2.imwrite(os.path.join(miss_path, im_name), img)

        lock.acquire()
        if cat_name not in cat_count.keys():
            cat_count[cat_name] = 1
        else:
            cat_count[cat_name] += 1 # 类别数+1 # ! 以图片为单位计数的
        lock.release()

        for detection in detections:
            x_rel, y_rel, w_rel, h_rel = detection['bbox']    
            conf = detection['conf']
            is_animal = detection['category']
            if is_animal == '2': # 为人
                continue
            if conf < 0.6:
                continue
   
            img_height, img_width, _ = img.shape
            x = int(x_rel * img_width)
            y = int(y_rel * img_height)
            w = int(w_rel * img_width)
            h = int(h_rel * img_height)

            if w<=0 or h<=0:
                continue

            x2 = x + w
            y2 = y + h
            if x < 0 or x2 >= img_width or y < 0 or y2 >= img_height:
                continue

            cropped = img.copy()[y:y2, x:x2]

            try:
                c_name = cat_name.replace(' ', '_') # ! 注意替换category空格为_，用于建立目录名
                crop_path = os.path.join(SAVE_ROOT, 'train_crop', c_name)
                if not os.path.exists(crop_path):
                    os.makedirs(crop_path)

                lock.acquire()
                if cat_name not in class_count.keys():
                    class_count[cat_name] = 1
                else:
                    class_count[cat_name] += 1
                lock.release()

                crop_name = c_name + '_' + str(class_count[cat_name]).zfill(MAX_COUNT) + '.jpg' # 类别名_00xxxx
                save_path = os.path.join(crop_path, crop_name)
                # print(save_path)
                cv2.imwrite(save_path, cropped)
            except:
                print("something wrong with: {}, bbox: {}".format(im_path, (x, y, x2, y2)))
                wrong_file_path = os.path.join(SAVE_ROOT, 'wrong_file')
                if not os.path.exists(wrong_file_path):
                    os.makedirs(wrong_file_path)
                lock.acquire()
                with open(os.path.join(wrong_file_path, wrong_file), 'a') as f:
                    f.write(im_path + '\t' + str((x, y, x2, y2)) + '\n')
                lock.release()
                continue
    print('Process %d Time-cost %.0f min' % (i, (time() - t0) / 60))

# 每个image_id一个目录，里面是小图
def CropTestImg(i, test_count, crop_count, lock):
    wrong_file = 'wrong_file.txt'
    t0 = time()
    start = test_length * i
    end = min(test_length * (i + 1), test_items)
    print("start process - {}, deal [{}, {}] data".format(i, start, end))
    cur_img_names = test_img_names[start: end]
    cur_img_id = test_img_id[start: end]

    for im_name, id in tqdm(zip(cur_img_names, cur_img_id), total=len(cur_img_names)):

        im_path = os.path.join(DATA_ROOT, 'test', im_name)
        img = cv2.imread(im_path)
        detections = megadetector_results_df[megadetector_results_df.id == id].detections.values[0]

        dir_path = os.path.join(TEST_SAVE_ROOT, id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        lock.acquire()
        if id not in crop_count.keys(): # 建立该image_id到crop_count
            crop_count[id] = 0
        lock.release()
        if len(detections) == 0: # train.json的标签有类别，但是detector的结果是没检测有box
            tmp = {'id': '', 'original_image_id': id}
            lock.acquire()
            test_count.append(tmp)
            lock.release()
            continue
        for detection in detections:
            x_rel, y_rel, w_rel, h_rel = detection['bbox']    
            conf = detection['conf']
            is_animal = detection['category']

            if is_animal == '2': # 为人
                continue
            if conf < 0.6:
                continue
   
            img_height, img_width, _ = img.shape
            x = int(x_rel * img_width)
            y = int(y_rel * img_height)
            w = int(w_rel * img_width)
            h = int(h_rel * img_height)

            if w<=0 or h<=0:
                continue

            x2 = x + w
            y2 = y + h
            if x < 0 or x2 >= img_width or y < 0 or y2 >= img_height:
                continue

            cropped = img.copy()[y:y2, x:x2]

            try:
                lock.acquire()
                crop_name = id + '_' + str(crop_count[id]).zfill(MAX_COUNT) + '.jpg' # id_0xx
                lock.release()
                save_path = os.path.join(dir_path, crop_name)
                # print(save_path)
                cv2.imwrite(save_path, cropped)
                # 成功保存一个crop
                tmp = {'id': crop_name, 'original_image_id': id}
                lock.acquire()
                test_count.append(tmp)
                crop_count[id] += 1
                lock.release()
            except:
                print("something wrong with: {}, bbox: {}".format(im_path, (x, y, x2, y2)))
                wrong_file_path = os.path.join(TEST_ROOT, 'wrong_file')
                if not os.path.exists(wrong_file_path):
                    os.makedirs(wrong_file_path)
                lock.acquire()
                with open(os.path.join(wrong_file_path, wrong_file), 'a') as f:
                    f.write(im_path + '\t' + str((x, y, x2, y2)) + '\n')
                lock.release()
                continue
        # 若遍历所有bbox, crop count[id]仍为0，则表明没有符合条件的小图，则test_count添加一个id='', original_image_id=id的字段
        lock.acquire()
        if crop_count[id] == 0:
            tmp = {'id': '', 'original_image_id': id}
            test_count.append(tmp)
        lock.release()
    print('Process %d Time-cost %.0f min' % (i, (time() - t0) / 60))

def crop_train_val():
    # 以图片为单位
    # cat_count:该类别的图片在标注里有多少个,
    # miss_cat_count:该图片在detector里面没有box出现，则miss了一张该类别的图片
    # 最后可以查看cat_count有的keys，但是class_count里没有的，则表明缺失的类别
    # class_count仅用于box为单位的图片命名计数

    # cat_count = defaultdict(int)
    # miss_cat_count = defaultdict(int)
    # class_count = defaultdict(int) # !以box为单位计数，用于命名
    wrong_file = 'wrong_file.txt'
    t1 = time()
    p = Pool(NUM_PROCESS)

    for i in range(NUM_PROCESS):
        p.apply_async(CropTrainImg, args=(i, share_cat_count, share_miss_cat_count, share_class_count, lock))
    p.close()
    p.join()

    print("All Process Over...")
    print('Total Time-cost %.0f min' % (time() - t1) / 60)
    print("Start to dump jsons")
    print("number of key-value in [cat_count.json] is {}".format(len(share_cat_count.items())))
    print("number of key-value in [class_count.json] is {}".format(len(share_class_count.items())))
    print("number of key-value in [miss_cat_count.json] is {}".format(len(share_miss_cat_count.items())))

    cat_count_json = json.dumps(dict(share_cat_count)) # !把DictProxy转成dict才能被JSON序列化
    class_count_json = json.dumps(dict(share_class_count))
    miss_cat_count_json = json.dumps(dict(share_miss_cat_count))
    if not os.path.exists(os.path.join(SAVE_ROOT, 'json_file')):
        os.makedirs(os.path.join(SAVE_ROOT, 'json_file'))
    with open(os.path.join(SAVE_ROOT, 'json_file', 'cat_count.json'), 'w') as json_file1:
        json_file1.write(cat_count_json)
    print("Save json file: cat_count.json")
    with open(os.path.join(SAVE_ROOT, 'json_file', 'class_count.json'), 'w') as json_file3:
        json_file3.write(class_count_json)
    print("Save json file: class_count.json")
    with open(os.path.join(SAVE_ROOT, 'json_file', 'miss_cat_count.json'), 'w') as json_file4:
        json_file4.write(miss_cat_count_json)
    print("Save json file: miss_cat_count.json")

# TODO 卡多少阈值，先取一个0.6的阈值，反正分类器会把背景类也预测出来
def crop_test():
    t1 = time()
    p = Pool(NUM_PROCESS)
    for i in range(NUM_PROCESS):
        p.apply_async(CropTestImg, args=(i, share_test_count, share_crop_count, lock))
    p.close()
    p.join()
    print("All Process Over...")
    print('Total Time-cost %.0f min' % ((time() - t1) / 60) )
    print("Start to dump jsons")
    print("number of key-value in [test_crop_count.json] is {}, Need number: {}".format(len(share_crop_count.items()), test_items))# 62894, 62894
    print("number of dicts in [test.json] is {}".format(len(share_test_count))) # 73972 所有小图的个数
    crop_count_json = json.dumps(dict(share_crop_count))
    with open(os.path.join(TEST_ROOT, 'test_crop_count.json'), 'w') as json_file:
        json_file.write(crop_count_json)
    print("Save json file: test_crop_count.json")   
    test_count_json = json.dumps(list(share_test_count))
    with open(os.path.join(TEST_ROOT, 'test.json'), 'w') as json_file:
        json_file.write(test_count_json)
    print("Save json file: test.json")

if __name__=='__main__':
	crop_train_val()
    crop_test()