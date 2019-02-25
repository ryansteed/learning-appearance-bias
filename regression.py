import argparse
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
import os
import pickle
import enlighten

from retrain import create_image_lists


def regress(*args, **kwargs):
    features = FeatureExtractor(*args, **kwargs).get_features()
    print(features)


class FeatureExtractor:
    def __init__(self, image_dir, testing_percentage, validation_percentage):
        self.image_dir = image_dir
        self.testing_percentage = testing_percentage
        self.validation_percentage = validation_percentage

    def get_features(self):
        try:
            return pickle.load(open(self.make_feature_name()))
        except FileNotFoundError:
            images_by_dir = create_image_lists(self.image_dir, self.testing_percentage, self.validation_percentage)
            features_by_image = {}

            manager = enlighten.get_manager()
            ticker = manager.counter(
                total=sum([len(val['training']) for key, val in images_by_dir.items()]),
                desc='Images Bottlenecked',
                unit='images'
            )
            for key, val in images_by_dir.items():
                for image in val['training']:
                    print(self.image_dir)
                    print(os.path.join(self.image_dir, key, image))
                    features_by_image[image] = extract_features_keras(os.path.join(self.image_dir, key, image))
                    ticker.update()
            pickle.dump(features_by_image, open(self.make_feature_name(), 'wb'))
            return features_by_image

    def make_feature_name(self):
        return 'models/features_{}'.format(self.image_dir)


def extract_features_keras(image_path):
    base_model = InceptionV3()
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predictions = model.predict(x)
    return np.squeeze(predictions)


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
        '--testing_percentage',
        type=float,
        default=0.2,
        help='Percentage of training set to use for testing.'
    )
    parser.add_argument(
        '--validation_percentage',
        type=float,
        default=0.1,
        help='Percentage of training set to hold out for validation.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    regress(**vars(FLAGS))
