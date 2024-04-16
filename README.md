


## Introduction
This project shows the method to detect fibres centre points from images at various noise levels. We have used four images from different dataset

1. Mock and UD - slice number 500 [link here](https://zenodo.org/records/5483719)
2. Wet 244p1 dataset - slice number 0999: data/Recons/Bunch2WoPR/rec20160318_223946_244p1_1p5cm_cont__4097im_1500ms_ML17keV_7.h5 [data link](http://dx.doi.org/doi:10.18126/M2QM0Z)
3. T700 reference - slice number 361 : T700-T-21_pco_0p4um_reference.h5 [data link](https://doi.org/10.5281/zenodo.7632124)

### Prerequisites
1. Install [PyTorch](https://pytorch.org/)  
   For the previous PyTorch versions, please check [here](https://pytorch.org/get-started/previous-versions/).
2. Install [JupyterNotebook or JupyterLab](https://jupyter.org/install)
3. Install requirements:  
- pip install -r requirements.txt

### Check the installation
- To check if the PyTorch GPU version has been installed successfully, run the following commands in Python:
```bash
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
```
### Data preparation
Prepare noisy data
```bash
data_prep.ipynb
```

### Training
Train a model from scratch using 2D slices.
```bash
train_fibre_mock.py
```

### Segment slices and performance evaluation
Perform segmentation on any specified 2D slice.
```bash
segm_fibre_mock.ipynb
```

## How UnetID-FibreSeg works
  1. The first step is to create a mask for the training image and crop them to the specified image size.
  2. The second step is to train the U-Net-id model using paired grayscale and manually labelled images.
  3. The final step is to use the trained U-Net-id model to segment other datasets.  
This is the flowchart: 
![flowchart](images/flowchart.png)
