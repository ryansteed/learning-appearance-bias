from appearance_bias.utils import Ticker

from PIL import Image
import pickle
import os
import pandas as pd
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import load_model

# import openface


class FeatureExtractor:
    """
    Class for extracting and caching features from image directories.
    """
    def __init__(self, image_dir, cache=True):
        """
        :param image_dir: Input image directories.
        :param cache: Whether or not to cache extracted features.
        """
        self.image_dir = image_dir
        self.cache = cache
        # TODO: make model choice a CLI flag
        self.model = FaceNetExtractionModel()
        # self.model = InceptionExtractionModel()

    def get_features(self, verbose=False):
        """
        Extract and export features from images in `image_dir`.
        :param verbose: Whether or not to print status updates
        :return: dataframe of features with file name AS 'Face name' and source directory AS 'Source'
        """
        features_by_image = []
        ticker = Ticker(
            len([file for subdir, dirs, files in os.walk(self.image_dir) for file in files]),
            'Images Bottlenecked',
            'images',
            verbose=verbose
        )
        # walk through images in all directories and subdirectories
        for subdir, dirs, files in os.walk(self.image_dir):
            for image in filter_images(files):
                # try loading from cache
                try:
                    features = pickle.load(open(self.make_feature_name(image, subdir), 'rb'))
                    if self.cache:
                        if verbose:
                            print("Loaded {} from file".format(image))
                    else:
                        raise FileNotFoundError
                # otherwise extract features
                except FileNotFoundError:
                    features = self.extract_features(os.path.join(subdir, image))
                    pickle.dump(features, open(self.make_feature_name(image, subdir), 'wb'))
                    if verbose:
                        print("Dumped {} to file".format(image))
                features_by_image.append(self.make_feature_dict(image, subdir, features))
                ticker.update()
        ticker.close()
        return pd.DataFrame(features_by_image).rename(columns={0: 'Face name', 1: 'Source'})

    @staticmethod
    def make_feature_dict(image, subdir, features):
        """
        Util function for constructing tuples for the feature dataframe.
        :param image: the image path
        :param subdir: the current parent directory of the image
        :param features: the vector of features corresponding to this image
        :return: a tuple
        """
        return [os.path.splitext(image)[0], os.path.basename(subdir)] + list(features)

    def make_feature_name(self, image, key):
        return 'models/features/{}__{}__{}.pkl'.format(
            os.path.basename(self.image_dir),
            os.path.basename(key),
            os.path.splitext(image)[0]
        )

    def extract_features(self, image_path):
        """
        Extract features using pre-set feature extraction model
        :param image_path: the image to featurize
        :return: the extracted features as a vector
        """
        return self.model.extract_features(image_path)


class ExtractionModel:
    """
    Class for extracting features from an image.
    """
    def extract_features(self, image_path):
        """
        Extract numeric features from an image.
        :param image_path: path to the image
        :return: features as a vector
        """
        raise NotImplementedError


class InceptionExtractionModel(ExtractionModel):

    def __init__(self):
        """
        Extracts features using an Inception architecture.
        """
        # specify these variables in inheritors
        self.model = None

    def extract_features(self, image_path):
        # load the image for TF parsing
        img = image.load_img(image_path, target_size=self.target_size)
        # convert pixels to array
        x = image.img_to_array(img)
        # expand to 3D
        x = np.expand_dims(x, axis=0)
        # pre-process for Inception
        x = preprocess_input(x)
        # predict using extraction model
        predictions = self.model.predict(x)
        return np.squeeze(predictions)


class ImageNetExtractionModel(InceptionExtractionModel):
    """
    Extracts features using Keras InceptionV3 (pre-trained on ImageNet ILSVRC images).
    """
    def __init__(self):
        super().__init__()
        base_model = InceptionV3()
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
        self.target_size = (299, 299)


class FaceNetExtractionModel(InceptionExtractionModel):
    """
    Extracts features using FaceNet (pre-trained on MS-Celeb-1M).
    """
    def __init__(self):
        super().__init__()
        self.model = load_model('models/facenet-keras/facenet_keras.h5', compile=False)
        self.target_size = (160, 160)


class LabelLoader:
    """
    Class for loading the labels for images.
    """
    base_labels = [
        "Attractive",
        "Competent",
        "Trustworthy",
        "Dominant",
        "Extroverted",
        "Likeable",
        "Threatening"
    ]
    labels = base_labels + [
        "Mean",
        "Frightening"
    ]
    label_mapping = {
        "Attractiveness": "Attractive",
        "Trustworthiness": "Trustworthy",
        "Competence": "Competent",
        "Dominance": "Dominant",
        "Extroverted": "Extroverted",
        "Likeable": "Likeable",
        "Threatening": "Threatening"
    }

    def __init__(self, image_dir):
        self.image_dir = image_dir

    def get_labels(self, **kwargs):
        """
        Get labels using CSV file if included, otherwise parse from filename.
        """
        try:
            return self.get_labels_csv(**kwargs)
        except FileNotFoundError:
            return self.get_labels_filename()

    def get_labels_csv(self, normalization=True):
        """
        Get labels from included CSV file.
        """
        df = pd.read_csv(self.make_label_filename())
        df = df[["Face name"] + [label for label in LabelLoader.labels if label in df.columns]]
        if normalization:
            for label in LabelLoader.labels:
                # z-score scaling (standard scaling)
                df[label] = (df[label] - df[label].mean()) / df[label].std(ddof=0) * 100
                # min-max scaling - scale to 0,1, then scale to -300, 300
                # df[label] = (df[label] - df[label].min()) / (df[label].max() - df[label].min()) * 600 - 300
        return df

    def get_labels_filename(self):
        """
        Get labels from the image filename. (Hardcoded for Todorov maximally distinct faces.)
        """
        labels = []
        for subdir, dirs, files in os.walk(self.image_dir):
            for image in filter_images(files):
                name = os.path.splitext(image)[0].split("_")
                labels.append({
                    "Face name": os.path.splitext(image)[0],
                    LabelLoader.label_mapping[name[-2].replace(' (300 faces)', '')]: float(name[-1])
                })
        return pd.DataFrame.from_records(labels)

    def make_label_filename(self):
        filename = os.path.join(self.image_dir, "labels.csv")
        try:
            open(filename)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Make sure that there is a CSV of emotion labels in the image directory {} named labels.csv".format(
                    self.image_dir
                )
            )
        return filename


def filter_images(files):
    """
    Util function for only considering image files with correct file format.
    :param files: files in a directory
    :return: the subset of `files` that are images
    """
    return [file for file in files if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png")]
