'''
This file is used to save the output image
'''

import torch.utils.data
from utils.parser import get_parser_with_args
from utils.helpers import get_test_loaders, initialize_metrics
import os
from tqdm import tqdm
import cv2
import time
from main_model import EnsembleModel

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_loader = get_test_loaders(opt, batch_size=1)



weight_path = './runs/train/'

##取weight_path第几个/后的文件名做为存储文件夹
#save_path = weight_path.split('/')[1].split('-')[0]
save_path = weight_path.split('/')[4]

base_path = './output_imgs/LEVIR-CD/'+save_path+'/'

if not os.path.exists(base_path):
    os.mkdir(base_path)

model = EnsembleModel(opt.ckp_paths, dev, input_size=opt.input_size)

model.eval()
index_img = 1
test_metrics = initialize_metrics()
with torch.no_grad():
    tbar = tqdm(test_loader)
    for batch_img1, batch_img2, labels, paths in tbar:

        img_paths = str(paths)
        img_paths = img_paths[2:]
        img_paths = img_paths[:-3]
        
        
        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        cd_preds = model(batch_img1,batch_img2)


        cd_preds = cd_preds[-1]

        cd_preds = cd_preds.data.cpu().numpy()
        cd_preds = cd_preds.squeeze() * 255


        file_path1 = str(base_path) + str(index_img) + 'gt' + '.png'
        cv2.imwrite(file_path1, cd_preds)
        if(index_img  == 1): 
            print(file_path1)


        index_img += 1

