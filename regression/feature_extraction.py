from regression.utils import Ticker

import pickle
import os
import pandas as pd
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model


class FeatureExtractor:
    def __init__(self, image_dir, cache=True):
        self.image_dir = image_dir
        self.cache = cache

        base_model = InceptionV3()
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

    def get_features(self, verbose=False):
        features_by_image = []

        ticker = Ticker(
            len([file for subdir, dirs, files in os.walk(self.image_dir) for file in files]),
            'Images Bottlenecked',
            'images'
        )

        for subdir, dirs, files in os.walk(self.image_dir):
            for image in filter_images(files):
                try:
                    features = pickle.load(open(self.make_feature_name(image, subdir), 'rb'))
                    if self.cache:
                        if verbose:
                            print("Loaded {} from file".format(image))
                    else:
                        raise FileNotFoundError
                except FileNotFoundError:
                    features = self.extract_features_keras(os.path.join(subdir, image))
                    pickle.dump(features, open(self.make_feature_name(image, subdir), 'wb'))
                    if verbose:
                        print("Dumped {} to file".format(image))
                features_by_image.append(self.make_feature_dict(image, subdir, features))
                ticker.update()
        ticker.close()
        return pd.DataFrame(features_by_image).rename(columns={0: 'Face name', 1: 'Source'})

    @staticmethod
    def make_feature_dict(image, subdir, features):
        return [os.path.splitext(image)[0], os.path.basename(subdir)] + list(features)

    def make_feature_name(self, image, key):
        return 'models/features/{}__{}__{}.pkl'.format(
            os.path.basename(self.image_dir),
            os.path.basename(key),
            os.path.splitext(image)[0]
        )

    def extract_features_keras(self, image_path):
        img = image.load_img(image_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        predictions = self.model.predict(x)
        return np.squeeze(predictions)


class LabelLoader:
    base_labels = [
        "Attractive",
        "Competent",
        "Trustworthy",
        "Dominant",
        "Extroverted",
        "Likeable"
    ]
    labels = base_labels + [
        "Mean",
        "Frightening",
        "Threatening",
    ]
    label_mapping = {
        "Attractiveness": "Attractive",
        "Trustworthiness": "Trustworthy",
        "Competence": "Competent",
        "Dominance": "Dominant",
        "Extroverted": "Extroverted",
        "Likeable": "Likeable"
    }

    def __init__(self, image_dir):
        self.image_dir = image_dir

    def get_labels(self):
        try:
            return self.get_labels_csv()
        except FileNotFoundError:
            return self.get_labels_filename()

    def get_labels_csv(self):
        df = pd.read_csv(self.make_label_filename())
        df = df[["Face name"] + LabelLoader.labels]
        for label in LabelLoader.labels:
            df[label] = (df[label] - df[label].mean()) / df[label].std(ddof=0) * 100
        return df

    def get_labels_filename(self):
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
    return [file for file in files if file.endswith(".jpg") or file.endswith(".jpeg")]
