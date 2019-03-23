import argparse
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
import os
import pickle
import enlighten
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score


def regress(image_dir, test_dir=None, cross_validate=True):
    print("Extracting training features...")
    features_train = FeatureExtractor(image_dir).get_features()
    print(features_train.describe())

    print("Extracting labels...")
    labels = LabelLoader(image_dir).get_labels()
    print(labels.describe())

    print(features_train["Face name"])
    print(labels["Face name"])
    df = pd.merge(
        labels,
        features_train,
        on="Face name",
        how="inner"
    )
    df.to_csv("output/data.csv")
    print(df.head())

    test = Regressor(df, "Trustworthy")

    if cross_validate:
        print("Cross validating...")
        test.cross_validate()

    print("Fitting model...")
    test.fit(split=False)

    if test_dir is not None:
        print("Extracting test features...")
        features_test = FeatureExtractor(
            test_dir
        ).get_features()
        print(features_test.describe())
        # print(test.predict(features_test))


class Regressor:
    def __init__(self, df, label):
        df = df[df[label].notna()]
        self.X = df.drop(LabelLoader.base_labels + ['Face name'], axis=1)
        self.y = df[label]
        self.reg = RandomForestRegressor(n_estimators=100)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.split()

    def split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=.2, random_state=42)

    def fit(self, split=True):
        if split:
            self.reg.fit(self.X_train, self.y_train)
        else:
            self.reg.fit(self.X, self.y)

    def cross_validate(self):
        print(cross_val_score(self.reg, self.X, self.y, cv=15).mean())

    def predict(self, X):
        return self.reg.predict(X)


class FeatureExtractor:
    def __init__(self, image_dir, cache=True):
        self.image_dir = image_dir
        self.cache = cache

        base_model = InceptionV3()
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

    def get_features(self, verbose=False):
        features_by_image = []

        manager = enlighten.get_manager()
        ticker = manager.counter(
            total=len([file for subdir, dirs, files in os.walk(self.image_dir) for file in files]),
            desc='Images Bottlenecked',
            unit='images'
        )
        for subdir, dirs, files in os.walk(self.image_dir):
            for image in filter_images(files):
                try:
                    if self.cache:
                        features_by_image.append(
                            self.make_feature_dict(
                                image,
                                pickle.load(open(self.make_feature_name(image, subdir), 'rb'))
                            )
                        )
                        if verbose:
                            print("Loaded {} from file".format(image))
                    else:
                        raise FileNotFoundError
                except FileNotFoundError:
                    features_by_image.append(
                        self.make_feature_dict(
                            image, self.extract_features_keras(os.path.join(subdir, image))
                        )
                    )
                    pickle.dump(features_by_image[-1][1:], open(self.make_feature_name(image, subdir), 'wb'))
                    if verbose:
                        print("Dumped {} to file".format(image))
                ticker.update()
        ticker.close()
        return pd.DataFrame(features_by_image).rename(columns={0: 'Face name'})

    @staticmethod
    def make_feature_dict(image, features):
        return [image] + list(features)

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
        return df[["Face name"] + LabelLoader.labels]

    def get_labels_filename(self):
        labels = []
        for subdir, dirs, files in os.walk(self.image_dir):
            for image in filter_images(files):
                name = os.path.splitext(image)[0].split("_")
                labels.append({
                    "Face name": image,
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type=str,
        default='',
        required=True,
        help='Path to folders of labeled images.'
    )
    parser.add_argument(
        '--test_dir',
        type=str,
        required=False,
        help='Path to folders of images to test on.'
    )
    parser.add_argument(
        '--no_validate',
        '-v',
        dest='cross_validate',
        action='store_false',
        help='Whether or not to cross validate the model.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)
    regress(**vars(FLAGS))
