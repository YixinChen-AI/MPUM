#!/bin/bash

# Navigate to nnUNet directory
cd nnUNet-1.7.1

# Install nnUNet
pip install -e .

# Go back to the parent directory
cd ../

# Install dependencies from requirements.txt
pip install numpy
pip install tqdm
pip install monai==1.2.0
pip install SimpleITK==2.2.1
