# MPUM
## Usage
### step 1: installation
```
git clone git@github.com:YixinChen-AI/MPUM.git
cd MPUM
chmod 777 ./install.sh
./install.sh
```
You need to install Torch according to your CUDA version. Torch official website: https://pytorch.org/

### step 2: well-trained ckpt
download well-trained .pth file
1. v0-1.pth
   - BaiDuWangPan: https://pan.baidu.com/s/1R_tyNTdgXdVaIEL9xRvaJQ  提取码：99x7 
   - Google Drive: Soon

## step 3: inference
1. You could use in .py
```
from inference import inference
config = {
                "tissuenumber":215,
                "modality":"ct",
                "modelsize":"base",
                "modalitydimension":512,
                "ckpt":f"../model_weight/SPC_base_512/epoch_{e}.pth",
            }
inference(config,
         nii_path=XXX,
          output_seg_path=XXX)
- config:
  - modality: "ct", "pet", "mr";
  - ckpt: the .pth path in step 2;
- nii_path: input nii path (not supported dcm files as input for now);
- output_set_path: the output dir path.
```

### step 4: check output
The output file structure:
```
output_seg_path/
├── readme.json # including the mapping between index and categories
└── merge.nii.gz # the segments for nii_path
```
