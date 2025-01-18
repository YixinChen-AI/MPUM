import torch,monai
import sys,os,json
import SimpleITK as sitk
import numpy as np
import torch.nn as nn
sys.path.append(".")
from datautils import resampleVolume,adjust_image_direction,resample_image
from inference import inference
from tqdm import tqdm
import pandas as pd
import sys
from inference import inference
config = {
                "tissue":"all",
                "modality":"ct",
                "modelsize":"base",
                "modalitydimension":512,
                "ckpt":["./weights/fold0.pth",
                        "./weights/fold1.pth",
                        "./weights/fold2.pth",]
            }
inference(config,
         nii_path="./samples/sample_pet.nii.gz",
         output_seg_path="./samples/output_pet_demo")