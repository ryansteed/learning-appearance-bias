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
from scipy.stats import ttest_ind


def regress(image_dirs, test_dir=None, cross_validate=False):
    reg = get_regressor("models/regressor.pkl", image_dirs)

    if cross_validate:
        print("Cross validating...")
        reg.cross_validate()

    if test_dir is not None:
        print("Extracting test features...")
        features_test = FeatureExtractor(
            test_dir
        ).get_features()
        print(features_test.describe())
        print("Predicting test images...")
        pred = reg.predict(features_test)
        features_test["pred"] = pred
        features_test.to_csv("output/pred.csv")
        print(features_test[["Source", "pred"]])
        print(features_test.groupby("Source").mean())
        print(features_test.groupby("Source").std())
        cats = np.unique(features_test["Source"])
        print(cats)
        print(ttest_ind(
            features_test[features_test["Source"] == cats[0]].pred,
            features_test[features_test["Source"] == cats[1]].pred
        ))


def get_regressor(filename, image_dirs):
    try:
        return pickle.load(open(filename, 'rb'))
    except (FileNotFoundError, EOFError):
        concats = []
        for image_dir in image_dirs:
            print("Extracting training features for {}...".format(image_dir))
            features_train = FeatureExtractor(image_dir).get_features()
            print(features_train.describe())

            print("Extracting labels for {}...".format(image_dir))
            labels = LabelLoader(image_dir).get_labels()

            df = pd.merge(
                labels,
                features_train,
                on="Face name",
                how="inner"
            )
            # print(df.columns)
            concats.append(df)

        df = pd.concat(concats, axis=0, join='inner', keys=[os.path.basename(image_dir) for image_dir in image_dirs])

        # print(df.describe())
        df.to_csv("output/data.csv")
        reg = Regressor(df, "Trustworthy")

        print("Fitting model...")
        reg.fit(split=False)

        pickle.dump(reg, open(filename, 'wb'))

        return reg


class Regressor:
    def __init__(self, df, label):
        df = df[df[label].notna()]
        self.label = label
        self.X, self.y = self.make_X_y(df)
        self.reg = RandomForestRegressor(n_estimators=100)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.split()

    def split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=.2,
            random_state=42
        )

    def fit(self, split=True):
        if split:
            self.reg.fit(self.X_train, self.y_train)
        else:
            self.reg.fit(self.X, self.y)
        return self.reg

    def cross_validate(self):
        print(cross_val_score(self.reg, self.X, self.y, cv=10).mean())

    def predict(self, test_df):
        X = self.make_X(test_df)
        return self.reg.predict(X)

    def make_X(self, df):
        to_drop = [x for x in LabelLoader.base_labels + ['Face name', 'Source'] if x in df.columns]
        return df.drop(columns=to_drop, axis=1)

    def make_X_y(self, df):
        return self.make_X(df), df[self.label]


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test_dir',
        type=str,
        required=False,
        help='Path to folders of images to test on.'
    )
    parser.add_argument(
        '--cross_validate',
        '-v',
        action='store_true',
        help='Whether or not to cross validate the model.'
    )
    parser.add_argument(
        '--image_dirs',
        type=str,
        nargs='+',
        default='',
        required=True,
        help='Path to folders of labeled images.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)
    regress(**vars(FLAGS))
