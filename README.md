# Modality-Projection Universal Model for Comprehensive Full-Body Medical Imaging Segmentation

[***ArXiv paper***](soon in some days)

The integration of deep learning in medical imaging has shown great promise for enhancing diagnostic, therapeutic, and research outcomes. However, applying universal models across multiple modalities remains challenging due to the inherent variability in data characteristics. This study aims to introduce and evaluate a Modality Projection Universal Model (MPUM). MPUM employs a novel modality-projection strategy, which allows the model to dynamically adjust its parameters to optimize performance across different imaging modalities. The MPUM demonstrated superior accuracy in identifying anatomical structures, enabling precise quantification for improved clinical decision-making. It also identifies metabolic associations within the brain-body axis, advancing research on brain-body physiological correlations. Furthermore, MPUM's unique controller-based convolution layer enables visualization of saliency maps across all network layers, significantly enhancing the model’s interpretability.

![image](https://github.com/YixinChen-AI/MPUM/blob/main/overview.png)
_Overview of the modality projection universal model. **a,** Training process of the MPUM leveraging data from three distinct modalities. **b,** Comparison of two common multimodal data training strategies with our proposed modality-projection strategy. **c,** Application of the MPUM model as an aided identification tool across three modalities (over 500 categories). **d,** The MPUM model is utilized as an computer-aided diagnosis (CAD) tool for precise localization of intracranial hemorrhage with CT scans. **e,** Application of the MPUM Model as an aided analysis tool in identifying altered metabolic correlations in regions affected by epilepsy. **f,** Additional experimental results, including t-SNE visualizations of feature extraction operators and analysis of the network's saliency map._

## Display of ICH results

![image](https://github.com/YixinChen-AI/MPUM/blob/main/case2.png)
_Performance of the MPUM Framework as aided diagnosis tool. **a,** Utilization of the MPUM as an aided diagnosis tool to detect hemorrhages and map brain regions from CT head scans, facilitating a precise diagnosis automatically. **b,** Illustration of the impact of the MPUM framework on enhancing diagnostic accuracy and support to general doctor in real-world settings._

## Display of Epilepsy results

![image](https://github.com/YixinChen-AI/MPUM/blob/main/qianfoshan.png)
_Multi-organ metabolic association analysis for pediatric epilepsy based on the universal model. We analyzed the metabolic associations in the epilepsy patient group (n=50) and the control group (n=22), using Fisher Z-Transformation to calculate the significance of differences in Pearson correlation coefficients. **a&b,** Schematic representation of the connectivity among brain regions associated with the Right Anterior Temporal Lobe Lateral Part and the right middle and inferior temporal gyrus. The left diagrams illustrate the strong metabolic connection within the control group. Notably, these correlations are statistically significantly reduced in the patient group (p<0.001). **c,** Metabolic connectivity between the Pallidum and Vertebrae T1-T12 affected by epilepsy._

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
import sys
sys.path.append({MPUM git path})
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
```
- config:
  - modality: "ct", "pet", "mr";
  - ckpt: the .pth path in step 2;
- nii_path: input nii path (not supported dcm files as input for now);
- output_set_path: the output dir path.


### step 4: check output
The output file structure:
```
output_seg_path/
├── readme.json # including the mapping between index and categories
└── merge.nii.gz # the segments for nii_path
```
