#将所有的图像随机分为Train,val,test三个数据集并且生成annotation
import os
import shutil
import random
import json

'''
# 将所有测试集图片复制到同一个文件夹下
img_root = '/home/ting/2022_bishe/data/HPI_Classification/test/'
img_dir = os.listdir(img_root)
for d in img_dir:
    # d 3类
    img_path = os.path.join(img_root,d)

    # tmp = ''

    # if d=='necrosis':
        # print(d)
        # tmp = 'A-'

    img_path1 = os.listdir(img_path)
    # 某一类的所有图片
    for d1 in img_path1:
        # print(img_path1)
        src = os.path.join(img_path,d1)
        # print('src')
        # print(src)
        dst = os.path.join('HPI_test_all', d + '_test_' + d1)
        # print('dst')
        # print(dst)
        if os.path.isfile(src):
            if not os.path.isfile(dst):
                shutil.copyfile(src, dst)
    
# quit()


####################################

# 将所有训练集图片复制到同一个文件夹下
img_root = '/home/ting/2022_bishe/data/HPI_Classification/train/'
img_dir = os.listdir(img_root)
for d in img_dir:
    # d 3类
    img_path = os.path.join(img_root,d)

    img_path1 = os.listdir(img_path)
    # 某一类的所有图片
    for d1 in img_path1:
        # print(img_path1)
        src = os.path.join(img_path,d1)
        # print('src')
        # print(src)
        dst = os.path.join('HPI_train_all', d + '_train_' + d1)
        # print('dst')
        # print(dst)
        if os.path.isfile(src):
            if not os.path.isfile(dst):
                shutil.copyfile(src, dst)
        # quit()
    
# quit()

'''

####################################

train_root = 'HPI_train_all'
trains = os.listdir(train_root)

test_root = 'HPI_test_all'
tests = os.listdir(test_root)


random.shuffle(trains)
random.shuffle(tests)

train_len = len(trains)
test_len = len(tests)
all_len = train_len + test_len
val_len = train_len*2//10

# train_list = trains
# val_list = imgs[train_len:val_len]
# print(trains)
train_list = trains[val_len:]
val_list = trains[:val_len]
# print(val_list)
test_list = tests

# all_img_dict = {'train':train_list,'val':val_list, 'test':test_list}
# all_img_dict = {'train':train_list, 'test':test_list}
all_img_dict = {'train':train_list, 'val':val_list, 'test':test_list}
print('train: ', len(train_list))
print('val: ', len(val_list))
print('test: ', len(test_list))
with open('heartTU_annotation.json','w') as j:
    json.dump(all_img_dict, j)

