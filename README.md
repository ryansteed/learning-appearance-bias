# Inception Retraining Script

## Installation
`git clone https://github.com/ryansteed/learning-appearance-bias`

Install Conda or Anaconda Python package manager. Also install `dlib` using [these instructions](https://github.com/ageitgey/face_recognition/issues/120):
```
wget https://bootstrap.pypa.io/get-pip.py
python get-pip.py
```

Then enter a virtual environment with the proper Python dependencies for scraping and analysis already configured from the `environment.yml` dependencies file:
```bash
conda env create -f environment.yml
source activate learning-appearance-bias
```

You may also need to install Tensorflow again if prompted. Make sure to do this from the newly created env.

`pip install tensorflow`

If your installation was interrupted by a Tensorflow installation error, resume the installation to complete building the environment.

`conda env update -f environment.yml`

You can also install the `appearance_bias` package locally with pip.

`pip install -e .`

## API

After environment installation, all endpoints can be accessed through the `main.py` CLI.

The CLI performs feature extraction, training and cross-validation of a regression model for predicting appearance bias for a given face.

For a quick summary of the CLI usage, just run:

`python main.py`

Feature extraction occurs automatically every time a new regression is trained and no features are available for the input dataset. 
**IMPORTANT**: If your images change, be sure to clear out the cached regression models in the `models` folder by running:

`rm models/*.pkl`

To produce features for and cross-validate a regression model on a particular trait (e.g. Trustworthy), run:
```
python main.py --image_dirs [path/to/training-images] [path/to/more-images] ... --label [Trait] -v
```

To produce features for and test a regression model on a particular trait (e.g. Trustworthy), run:
```
python main.py --image_dirs [path/to/training-images] [path/to/more-images] ... --label [Trait] --test_dir [path/to/test-images]
```

## Image Directories
Image directories must contain images in JPG format, pre-processed. Directories should not be nested - each directory 
passed in to the CLI will only be checked for images at the first level.

For the data used in our paper, labels for the maximally distinct images are automatically parsed from the file names.

For randomly generated or any other images, a file named `label.csv` must be included in the image directory with labels 
matched to each file name. 
For the 300 Random Faces used in our paper, the `labels.csv` file is provided in `random.zip`.

## Contents

|File|Description|
|---|---|
|`data/`| Directory containing provided data for reproducing figures, though data can be stored anywhere.|
|`models/`| contains pre-trained or cached models used/produced during training. Already includes a version of FaceNet pre-trained on MS-Celeb-1M from the [open-source FaceNet release](https://github.com/davidsandberg/facenet), converted for Keras using [keras-facenet](https://github.com/nyoki-mtl/keras-facenet).|
|`output/`| Automatically generated directory with output plots and features.|
|`appearance_bias/`| Python module containing source code for `main.py` CLI.|
|`scripts/`| Several useful and some deprecated scripts for image processing. `plot_preds.R` produces figures from regression output.|
|`environment.yml`|Dependencies file for conda env.|
|`main.py`| CLI script used to run regression training. See [API](#api) for full documentation.|
|`retrain.py`| *DEPRECATED* Tensorflow script for retraining an Inception model from scratch and classifying new images.|

