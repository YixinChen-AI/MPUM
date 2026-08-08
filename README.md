# Modality-Projection Universal Model for Comprehensive Full-Body Medical Imaging Segmentation

[***ArXiv paper***](https://arxiv.org/abs/2412.19026)

![MPUM Tutorial Video](https://github.com/YixinChen-AI/MPUM/blob/main/tutorial.gif)

> **Dynamic PET is supported through a validated adapter.** The adapter converts
> quantitative dynamic PET DICOM into a duration-weighted static SUVbw reference,
> which can then be segmented by the standard MPUM PET inference API. See the
> [Dynamic PET quick start](#dynamic-pet-quick-start).


# Table of Contents
- [Introduction](#introduction)
- [Display of ICH results](#display-of-ich-results)
- [Display of Epilepsy results](#display-of-epilepsy-results)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Usage](#usage)
  - [Dynamic PET quick start](#dynamic-pet-quick-start)
  - [Inference](#inference)
- [License](#license)

# Introduction
The integration of deep learning in medical imaging has shown great promise for enhancing diagnostic, therapeutic, and research outcomes. However, applying universal models across multiple modalities remains challenging due to the inherent variability in data characteristics. This study aims to introduce and evaluate a Modality Projection Universal Model (MPUM). MPUM employs a novel modality-projection strategy, which allows the model to dynamically adjust its parameters to optimize performance across different imaging modalities. The MPUM demonstrated superior accuracy in identifying anatomical structures, enabling precise quantification for improved clinical decision-making. It also identifies metabolic associations within the brain-body axis, advancing research on brain-body physiological correlations. Furthermore, MPUM's unique controller-based convolution layer enables visualization of saliency maps across all network layers, significantly enhancing the model’s interpretability.

![image](https://github.com/YixinChen-AI/MPUM/blob/main/overview.png)
_Overview of the modality projection universal model. **a,** Training process of the MPUM leveraging data from three distinct modalities. **b,** Comparison of two common multimodal data training strategies with our proposed modality-projection strategy. **c,** Application of the MPUM model as an aided identification tool across three modalities (over 500 categories). **d,** The MPUM model is utilized as an computer-aided diagnosis (CAD) tool for precise localization of intracranial hemorrhage with CT scans. **e,** Application of the MPUM Model as an aided analysis tool in identifying altered metabolic correlations in regions affected by epilepsy. **f,** Additional experimental results, including t-SNE visualizations of feature extraction operators and analysis of the network's saliency map._

# Display of ICH results

![image](https://github.com/YixinChen-AI/MPUM/blob/main/case2.png)
_Performance of the MPUM Framework as aided diagnosis tool. **a,** Utilization of the MPUM as an aided diagnosis tool to detect hemorrhages and map brain regions from CT head scans, facilitating a precise diagnosis automatically. **b,** Illustration of the impact of the MPUM framework on enhancing diagnostic accuracy and support to general doctor in real-world settings._

# Display of Epilepsy results

![image](https://github.com/YixinChen-AI/MPUM/blob/main/qianfoshan.png)
_Multi-organ metabolic association analysis for pediatric epilepsy based on the universal model. We analyzed the metabolic associations in the epilepsy patient group (n=50) and the control group (n=22), using Fisher Z-Transformation to calculate the significance of differences in Pearson correlation coefficients. **a&b,** Schematic representation of the connectivity among brain regions associated with the Right Anterior Temporal Lobe Lateral Part and the right middle and inferior temporal gyrus. The left diagrams illustrate the strong metabolic connection within the control group. Notably, these correlations are statistically significantly reduced in the patient group (p<0.001). **c,** Metabolic connectivity between the Pallidum and Vertebrae T1-T12 affected by epilepsy._

# System Requirements
## Hardware requirements
`MPUM` requires a standard computer with sufficient RAM and an NVIDIA GPU with more than 12 GB of memory.

## Software requirements
### OS Requirements
This package is supported for *Linux*. The package has been tested on the following systems:
+ Linux: Ubuntu 20.04, Rocky Linux

### Python Dependencies
`MPUM` mainly depends on the Python scientific stack.
```
numpy
tqdm
monai==1.2.0
SimpleITK==2.2.1
pydicom>=2.4
```
# Installation Guide


## Install from Github
```
git clone git@github.com:YixinChen-AI/MPUM.git
cd MPUM
chmod 777 ./install.sh
./install.sh
```
You need to install Torch according to your CUDA version. Torch official website: https://pytorch.org/


download well-trained .pth file
1. v0-1.pth
   - BaiDuWangPan: https://pan.baidu.com/s/1R_tyNTdgXdVaIEL9xRvaJQ  提取码：99x7 
   - Google Drive: Soon
## Install from Pypi
Soon
# Usage
## Dynamic PET quick start

MPUM segments a three-dimensional PET SUV image rather than a four-dimensional
time series. The dynamic PET adapter therefore creates a static late-window
SUVbw reference first; it does not discard or modify the source dynamic frames.

### 1. Build the static SUVbw reference

```bash
python -m dynamic_pet.cli /path/to/dicom /path/to/reference_output \
  --late-seconds 600 --case-key case_001
```

Supported layouts:

| Input layout | Description |
|---|---|
| Classic dynamic PET | One DICOM object per slice and time frame |
| Enhanced PET | One DICOM object containing a complete 3-D time frame |

The default command uses the final 600 seconds. Select a different window when
required by the tracer and acquisition protocol. Frames are selected by their
actual overlap with the requested window and averaged using overlap duration as
the weight. Per-frame/per-slice rescale values and physical LPS geometry are
preserved.

Reference output:

```text
reference_output/
├── pet_late_600s_suvbw.nii.gz
└── provenance.json                 # classic DICOM
```

Enhanced PET writes the same NIfTI plus `pet_late_600s_report.json`. Provenance
records frame timing, effective window coverage, series selection, decay
handling, SUV metadata, geometry, and excluded incomplete frames. Source DICOM
files are treated as read-only.

### 2. Run standard MPUM PET inference

```python
from inference import inference

config = {
    "tissue": "all",
    "modality": "pet",
    "modelsize": "base",
    "modalitydimension": 512,
    "ckpt": [
        "/path/to/fold0.pth",
        "/path/to/fold1.pth",
        "/path/to/fold2.pth",
    ],
    "correct_brain_laterality": True,
}

inference(
    config,
    nii_path="/path/to/reference_output/pet_late_600s_suvbw.nii.gz",
    output_seg_path="/path/to/segmentation_output",
)
```

The segmentation is written as `segmentation_output/merge.nii.gz` in the
original reference-image geometry.

### Input safety checks

Inputs are rejected when SUVbw cannot be derived safely—for example, missing
or zero patient weight/injected dose, non-BQML units, absent decay/attenuation/
scatter/dose-calibration declarations, ambiguous decay timing, or inconsistent
geometry. Incomplete time frames are excluded and recorded in provenance; the
adapter fails if no complete frames remain. After reference generation, pass
the resulting `pet_late_600s_suvbw.nii.gz` to the normal PET inference API.

These checks are intentional: guessing injected activity or converting
non-quantitative counts would change MPUM's PET intensity scale and can produce
misleading segmentation.

### Validation status and limitations

Technical validation covered both supported DICOM layouts. In a ten-case
classic-DICOM cohort, all six quantitatively valid cases completed MPUM
inference with matching output geometry; four invalid cases were rejected for
zero injected activity or non-BQML, non-attenuation/scatter-corrected data. The
Enhanced PET and classic-DICOM adapters also reproduced independently built
reference volumes exactly (maximum absolute voxel difference 0).

This establishes technical and visual feasibility, not clinical accuracy: the
cohort did not include manual segmentation ground truth. Partial-field-of-view
scans may also contain small anatomically implausible labels. The current MPUM
output includes label selection, brain-laterality correction, and resampling to
the input geometry; it does **not** apply largest-connected-component filtering
or other anatomical cleanup. Preserve `merge.nii.gz` as the raw model output if
adding application-specific post-processing.

The adapter currently accepts DICOM input, not 4-D NIfTI time series. Static
NIfTI input remains supported directly by the inference API below.

## inference
1. You could use in .py
```
import sys
sys.path.append({MPUM git path})
from inference import inference
config = {
                "tissue":"brain", # (str), "all", "brain"
                "modality":"pet", # (str), "ct", "pet", "mr"
                "modelsize":"base",
                "modalitydimension":512,
                # Correct the released checkpoints' bilateral brain channels.
                # Enabled by default; set False only to reproduce legacy output.
                "correct_brain_laterality": True,

                # single model mode
                "ckpt":"<the path of ckpt file which has downloaded above (.pth file)>",
                # multi model ensemble model
                "ckpt":["the first model ckpt path",
                        "the second model ckpt path",
                        ......]
            }
inference(config,
         nii_path=XXX,
         output_seg_path=XXX)
```
- config:
  - modality: "ct", "pet", "mr";
  - ckpt: the .pth path in step 2;
  - correct_brain_laterality: swaps the 40 bilateral brain channel pairs before
    label selection. This is enabled by default because the released checkpoints
    otherwise place all bilateral brain labels on the opposite physical side.
- nii_path: static NIfTI input path; use the dynamic PET adapter above for DICOM;
- output_seg_path: output directory path.

```
# Take CT scan as an example
inference(config,
         nii_path="./sample/sample_ct.nii.gz",
         output_seg_path="./sample/output_ct")
# Take PET scan as an example
inference(config,
         nii_path="./sample/sample_pet.nii.gz",
         output_seg_path="./sample/output_pet")
```
## check output
The output file structure:
```
output_seg_path/
├── readme.json # including the mapping between index and categories
└── merge.nii.gz # the segments for nii_path
```
# License
This project is covered under the **Apache 2.0 License**.
