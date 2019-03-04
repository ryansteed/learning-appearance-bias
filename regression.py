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
    def __init__(self, image_dir, testing_percentage, validation_percentage, cache=True):
        self.image_dir = image_dir
        self.testing_percentage = testing_percentage
        self.validation_percentage = validation_percentage
        self.cache = cache

        base_model = InceptionV3()
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

    def get_features(self):
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
                try:
                    if self.cache:
                        features_by_image[image] = pickle.load(open(self.make_feature_name(image), 'rb'))[1]
                    else:
                        raise FileNotFoundError
                except FileNotFoundError:
                    features_by_image[image] = self.extract_features_keras(os.path.join(self.image_dir, key, image))
                    pickle.dump((image, features_by_image[image]), open(self.make_feature_name(image), 'wb'))
                ticker.update()
        return features_by_image

    def make_feature_name(self, image):
        return 'models/features/{}_{}.pkl'.format(os.path.basename(self.image_dir), os.path.splitext(image)[0])

    def extract_features_keras(self, image_path):
        img = image.load_img(image_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        predictions = self.model.predict(x)
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
