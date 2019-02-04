# Inception Retraining Script

## Legacy README from Tensorflow
retrain.py is an example script that shows how one can adapt a pretrained
network for other classification problems. A detailed overview of this script
can be found at: https://www.tensorflow.org/tutorials/image_retraining

The script also shows how one can train layers
with quantized weights and activations instead of taking a pre-trained floating
point model and then quantizing weights and activations.
The output graphdef produced by this script is compatible with the TensorFlow
Lite Optimizing Converter and can be converted to TFLite format.

## Processing
I use the `face_recognition` Python package for face detection. The `scripts/crop.py` script iterates through any given folder, detecting faces in any `.jpg` files and creating new, cropped images for each face in the same directory.

```bash
python crop.py [path/to/dir]
```

`scripts/face_detect.sh` - a less useful script that performs the same task using a CSV containing each image name and its face coordinates, which can be obtained with `./face_detect.sh [path/to/dir/..]`

## Re-training
To retrain the basic Inception v3 model on any given new images, create a directory containing all the training images sorted into separate directories by class.

```bash
python retrain.py \
    --image_dir [path/to/repo_with_image_dirs] \
    --output_graph models/[training_set].pb \
    --output_labels models/[training_set].txt

```

## Testing
To make predictions on a set of images, use the `scripts/label_image.py` script. The `scripts/classify_dir.sh` script labels all the images in a directory and stores in a CSV file. Example:

```bash
./scripts/classify_dir.sh ../data/todorov/random/ /tmp/output_graph.pb /tmp/output_labels.txt
```

