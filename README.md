# "Learning" Appearance Bias

Software package for the paper https://arxiv.org/abs/2002.05636.

NOTE: This software is for scientific replication only and released under a non-commercial license. The models trained here are not fit for use in any application and are *only* intended for studying whether and to what extent traditional machine learning methods embed subjective human prejudices. To prevent unethical misuse, the training data are available only for replication, by request.

## Ethics
The code in this repository tests whether traditional ML techniques (in this case, FaceNet trained on faces annotated in a laboratory setting) are capable of embedding subjective appearance biases from human annotators. Researchers and companies often utilize computer vision models to predict similarly subjective personality attributes such as “employability" - but we showed that the ML techniques we investigated only embed deliberately manipulated/incorporated signals of bias, which provide neither an objective measure of personality traits nor a even a good measure of subjective bias itself. The methods we tested lack external validity (correlation with real-world outcomes and other cultural contexts) and internal validity (the ability to even explain annotations of randomly generated faces in a laboratory setting).

Still, recent papers ([Safra et al., 2020](https://www.nature.com/articles/s41467-020-18566-7); [Peterson et al., 2022](https://www.pnas.org/doi/10.1073/pnas.2115228119)) use newer ML techniques to attempt to build better models for predicting perception biases. These methods risk enabling applications that deliberately automate human biases (e.g. for a subjective hiring algorithm, like the ones [recently used by HireVue](https://www.washingtonpost.com/technology/2019/10/22/ai-hiring-face-scanning-algorithm-increasingly-decides-whether-you-deserve-job/)).

The software included here only illustrates the *limitations* of ML in this area (for more, see [this critique of Safra et al.](https://arxiv.org/abs/2202.08674) which cites our work). In response to continued efforts to use ML to automate appearance biases, we make the training data used (not our own) available only for replication purposes, on request. The rest of the code is left public under a non-commercial license for the sole purpose of helping readers understand and replicate our research.

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
|`data/`| Directory containing provided data for reproducing figures, though data can be stored anywhere. Available on request.|
|`models/`| contains pre-trained or cached models used/produced during training. Already includes a version of FaceNet pre-trained on MS-Celeb-1M from the [open-source FaceNet release](https://github.com/davidsandberg/facenet), converted for Keras using [keras-facenet](https://github.com/nyoki-mtl/keras-facenet).|
|`output/`| Automatically generated directory with output plots and features.|
|`appearance_bias/`| Python module containing source code for `main.py` CLI.|
|`scripts/`| Several useful and some deprecated scripts for image processing. `plot_preds.R` produces figures from regression output.|
|`environment.yml`|Dependencies file for conda env.|
|`main.py`| CLI script used to run regression training. See [API](#api) for full documentation.|
|`retrain.py`| *DEPRECATED* Tensorflow script for retraining an Inception model from scratch and classifying new images.|

