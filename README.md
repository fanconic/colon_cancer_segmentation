# Cancer Segmentation - ML4H Project1
Project 1 of the course Machine Learning for Health Care in FS21 at ETH Zürich

<img src="https://github.com/fanconic/ML4H_project1/blob/main/misc/u_net.png" width="500">

## Objective
The goal of this project was to build a machine learning model that segments CT scans according to voxels containing cancerous or non-cancerous tissue (i.e. masking the input as primary or background). Specifically, we are looking at **3D CT scans** which contain **colon cancer**. We implemented a **U-Net architecture**, with customized data loader for this image segmentation. Below find a plot of what our model does. We show a slice of the CT scan representing the input to the model, follwed by our prediction and the actual cancerous segmentation. Finally, for visualization purpose we show a mixture of the input and our prediction, as well as histogram of the prediction.

![Cancer Segmentation Example from our Model](https://github.com/fanconic/ML4H_project1/blob/main/misc/segment_example.png)

## Setup

### Installation

Clone this repository.
```bash
$ git clone https://github.com/fanconic/ML4H_project1
$ cd ML4H_project1
```

We suggest to create a virtual environment and install the required packages.
```bash
$ conda create -n test_env
$ conda activate test_env
$ pip install -r requirements.txt
```

### Dataset

Download the Dataset from https://drive.google.com/drive/folders/1Fb9RzgBPJAVFkqUD4gAjb94L-6qkAquJ?usp=sharing and extract the files in a suitable folder.

### Repository Structure

- `train.py`: Main training loop in PyTorch
- `predict.py`: Prediction script
- `cross-validation.py`: Script to elaborate on the most suited model configuration 
- `settings.py`: Settings for paths, hyperparameters and variables
- `run.sh`: Script to train the U-Net on Leonhard
- `run_predict.sh`: Run the prediction script on Leonhard

### Source Code Directory Tree
```
.
└── src                 # Source code for the experiments
    ├── data                # Data loaders, preprocessing
    ├── models              # U-Net model, metrics, losses
    └── utils               # Helper functions
├── log                 # Cross validation training log
└── misc                # Images for README
```


## How to run on the Leonhard Cluster:
```
module load eth_proxy python_gpu/3.7.4
pip install --user nibabel
pip install --user --upgrade torch
pip install --user --upgrade torchvision
```

To train the model:
```
bsub -n 4 -W HH:MM -N -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" ./run.sh
```

To predict:
```
bsub -n 4 -W HH:MM -N -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" ./run_predict.sh
```

## Prediction Files
The predictions for the given testing set (e.g. colon_{ID}.nii) are stored as pickle files, each containing the corresponding ID of the 3D image (e.g. prediction_colon_{ID}.pkl). The prediction has the shape height x width. x depth.


## Contributors
- Claudio Fanconi - fanconic@ethz.ch
- Manuel Studer - manstude@ethz.ch
- Severin Husmann - shusmann@ethz.ch

## References:
- https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
- https://www.kaggle.com/godeep48/simple-unet-pytorch
- https://en.wikipedia.org/wiki/Hounsfield_scale

