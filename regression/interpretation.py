from regression.feature_extraction import FaceNetExtractionModel

import numpy as np
import matplotlib.pyplot as plt

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries

from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image

from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVC



class Interpreter:
    
    def __init__(self, 
            clf=SVC(kernel='linear', probability=True), 
            segmenter=SegmentationAlgorithm('slic', n_segments=300, compactness=1, sigma=1),
            extraction_model = FaceNetExtractionModel()
        ):
        self.clf = clf
        self.segmenter = segmenter
        self.extraction_model = extraction_model

    def fit(self, X, y, val=True, continuous=True):
        if continuous:
            y = Interpreter.binarize(y)
        self.clf.fit(X, y)
        if val:
            cross_val_score(
                self.clf, X, y, 
                cv=KFold(n_splits=10, shuffle=True, random_state=42)
            ).mean()

    def predict_fn(self, img_arrays):
        embeddings = self.extraction_model.model.predict(img_arrays)
        preds = self.clf.predict_proba(embeddings)
        return preds

    # from https://marcotcr.github.io/lime/tutorials/Tutorial%20-%20images.html
    def explain_img(self, img, label, name="", ground_truth=None, save_path=None, verbose=False, **kwargs):
        img_processed = preprocess_input(image.img_to_array(img)).astype(float)
        sample_embedding = self.extraction_model.model.predict(np.array([img_processed]))
        pred = self.predict_fn(np.array([img_processed]))
        
        explainer = lime_image.LimeImageExplainer(verbose=False)
        if verbose:
            print("Image {} w/ ground truth val {}".format(name, ground_truth))
            print("Explaining label {} (p={}) for attribute {}".format(
                self.interpret_binary(np.argmax(pred)), pred[:,np.argmax(pred)], label)
            )
        default_kwargs = {
            'hide_color': 0, 
            'num_samples': 10000,
            'num_features': 100000
        }
        default_kwargs.update(kwargs)
        explanation = explainer.explain_instance(
            img_processed, 
            self.predict_fn,
            segmentation_fn=self.segmenter,
            **default_kwargs
        )
        temp, mask = explanation.get_image_and_mask(
            np.argmax(pred), 
            positive_only=False, 
            num_features=10, 
            hide_rest=False
        )
        
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(img_processed / 2 + 0.5)
        ax[1].imshow(mark_boundaries(temp / 2 + 0.5, mask))
        if verbose: plt.show()
        if save_path is not None: plt.savefig(save_path)

        return explanation

    def get_img(self, name, source, subd=None, d="data", file="jpg"):
        img_path = "{}/{}/{}/{}.{}".format(d, subd, source, name, file)
        if subd is None:
            img_path = "{}/{}/{}.{}".format(d, source, name, file)
        return image.load_img(img_path, target_size=self.extraction_model.target_size)

    @staticmethod
    def interpret_binary(y):
        return "Positive" if y else "Negative"

    @staticmethod
    def binarize(y, threshold=0):
        return (y > threshold).astype(int)