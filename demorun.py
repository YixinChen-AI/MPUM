import torch,monai
import sys,os,json
import SimpleITK as sitk
import numpy as np
import torch.nn as nn
sys.path.append("/share/home/yxchen/github/MPUM")
from datautils import resampleVolume,adjust_image_direction,resample_image
from inference import inference


from tqdm import tqdm
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"]='7'

import sys
sys.path.append("/share/home/yxchen/github/MPUM")
from inference import inference
config = {
                "tissue":"all",
                "modality":"ct",
                "modelsize":"base",
                "modalitydimension":512,
                "ckpt":["/share/home/yxchen/08TripleUniversalModel/model_weight/3fold_MPUM_base_512/fold1/epoch_43.pth",
                        "/share/home/yxchen/08TripleUniversalModel/model_weight/3fold_MPUM_base_512/fold2/epoch_42.pth",
                        "/share/home/yxchen/08TripleUniversalModel/model_weight/3fold_MPUM_base_512/fold0/epoch_40.pth"]
            }
inference(config,
         nii_path="/share/home/yxchen/github/MPUM/samples/sample_pet.nii.gz",
         output_seg_path="/share/home/yxchen/github/MPUM/samples/output_pet")