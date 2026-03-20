import os
import torch.utils.data as data
from PIL import Image
from utils import transforms as tr
import numpy as np

'''
Load all training and validation data paths加载所有训练和验证数据的路径
'''
def full_path_loader(data_dir):
    train_data = [i for i in os.listdir(data_dir + 'train/A/') if not
    i.startswith('.')]
    train_data.sort()

    valid_data = [i for i in os.listdir(data_dir + 'val/A/') if not
    i.startswith('.')]
    valid_data.sort()

    train_label_paths = []
    val_label_paths = []
    for img in train_data:
        train_label_paths.append(data_dir + 'train/OUT/' + img)
        # train_label_paths.append(data_dir + 'train/OUT/' + img)
    for img in valid_data:
        val_label_paths.append(data_dir + 'val/OUT/' + img)


    train_data_path = []
    val_data_path = []

    for img in train_data:
   
         train_data_path.append([data_dir + 'train/A/'+img, data_dir + 'train/B/'+img])#交换AB顺序
         # train_data_path.append([data_dir + 'train/B/'+img, data_dir + 'train/A/'+img])
         # 
    
      
    for img in valid_data:
      
        val_data_path.append([data_dir + 'val/A/'+img, data_dir + 'val/B/'+img])

    train_dataset = {}
    val_dataset = {}
    #训练时不需要路径，直接设置1占位
    for cp in range(len(train_data)):
        train_dataset[cp] = {'image': train_data_path[cp],
                         'label': train_label_paths[cp],
                         'path': 1}
    for cp in range(len(valid_data)):
        val_dataset[cp] = {'image': val_data_path[cp],
                         'label': val_label_paths[cp],
                         'path': 1}


    return train_dataset, val_dataset

'''
Load all testing data paths
'''
def full_test_loader(data_dir):

    test_data = [i for i in os.listdir(data_dir + 'test/A/') if not
                    i.startswith('.')]
    test_data.sort()

    #获取前后序图片
    test_data_path = []
    for img in test_data:
        test_data_path.append([data_dir + 'test/A/'+img, data_dir + 'test/B/'+img])
    
    #test_pic_paths列表只用第一位
    #测试时需要路径与图片一一对应
    test_dataset = {}
    for cp in range(len(test_data)):
        test_dataset[cp] = {'image': test_data_path[cp]}
    #print(test_dataset)

    return test_dataset

def cdd_loader(img_path, aug):
   
    name1=img_path[0]
    name2=img_path[1]
    img1 = Image.open(name1)
    img2 = Image.open(name2)
    # print(label_path)

    
    sample = {'image': (img1, img2)}
    

    if aug:
        sample = tr.train_transforms(sample)
    else:
        sample = tr.test_transforms(sample)

    #print(sample)
    
    return sample['image'][0], sample['image'][1]


class CDDloader(data.Dataset):
    def __init__(self, full_load, aug=False):

        self.full_load = full_load
        self.loader = cdd_loader
        self.aug = aug

    def __getitem__(self, index):
        img_path = self.full_load[index]['image']

        return self.loader(img_path,
                           self.aug)
    

    def __len__(self):
        return len(self.full_load)
