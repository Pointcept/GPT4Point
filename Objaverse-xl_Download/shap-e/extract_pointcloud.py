# ==============================================================================
# Copyright (c) 2023 Tiange Luo, tiange.cs@gmail.com
# Last modified: September 05, 2023
#
# This code is licensed under the MIT License.
# ==============================================================================

import torch
import time
import os
from shap_e.util.data_util import load_or_create_multimodal_batch
import argparse
import tqdm
import pickle
import random
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--mother_dir', type = str, default='/home/qizhangyang/rendering_objaverse_xl/render_save_1/renders') # path to store zip files for rendered images
parser.add_argument('--cache_dir', type = str, default='./cache_npz') # path to cache npz format
parser.add_argument('--save_name', type = str, default='extracted_pts') # path to output final pointcloud ply results
args = parser.parse_args()

device = 'cpu'

from shap_e.models.download import load_model
xm = load_model('transmitter', device=device)

target_dir = './%s'%args.save_name
os.makedirs(target_dir, exist_ok=True)

zip_files = glob.glob(f"{args.mother_dir}/*.zip")


with torch.no_grad():  
    for file_path in tqdm.tqdm(zip_files):
        print('Begin to extract point clouds:', file_path)
        try:
            pc = load_or_create_multimodal_batch(
                device,
                model_path= file_path, 
                mv_light_mode="basic",
                mv_image_size=256,
                pc_num_views=20,
                cache_dir=args.cache_dir,
                verbose=True, # This will show Blender output during renders
            )
            if not os.path.exists(os.path.join(target_dir, '%s.ply'%file_path.split('/')[-1].split('.')[0])) and pc is not None:
                os.popen('python npz2ply.py --npz_filename %s --ply_filename %s'%(os.path.join(args.cache_dir, 'pc_%s_mat_20_524288_16384.npz'%file_path.split('/')[-1]), os.path.join(target_dir, '%s.ply'%file_path.split('/')[-1].split('.')[0]))).read()

        except:
            print('Error:', file_path)
            continue



