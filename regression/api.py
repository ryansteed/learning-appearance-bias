from regression.feature_extraction import FeatureExtractor, LabelLoader
from regression.model import Regressor
from regression.interpretation import Interpreter
import os
import pickle
import pandas as pd


def regress(label, **kwargs):
    if label is None:
        return regress_all(**kwargs)
    return regress_single(label, **kwargs)


def regress_all(test_dir=None, **kwargs):
    labels = ["Attractive", "Competent", "Dominant", "Extroverted", "Likeable", "Trustworthy"]
    results = []
    for label in labels:
        results.append(regress_single(label, test_dir=test_dir, **kwargs))

    if all(result is not None for result in results):
        for i, result in enumerate(results):
            pred_name = "pred_{}".format(labels[i])
            result = result.rename(columns={"pred": pred_name})
            if i == 0:
                df = result
            else:
                df = merge_x_y(df, result[["Face name", pred_name]])
        df.to_csv("output/preds/{}-preds_all.csv".format(os.path.basename(test_dir)))


def regress_single(label, image_dirs, test_dir=None, cross_validate=False):
    reg = get_regressor(label, image_dirs)

    if cross_validate:
        print("Cross validating...")
        reg.cross_validate(label, mse=True, test_random=True)
        # reg.cross_validate(label, mse=True)
        reg.chart("{}_scatter".format(label), annotate=False)
        reg.chart("{}_scatter_folds".format(label), annotate=False, hue="fold")
        reg.chart("{}_scatter_annotated".format(label), annotate=True)

        return None

    if test_dir is not None:
        print("Extracting test features...")
        features_test = FeatureExtractor(
            test_dir
        ).get_features()

        print("Predicting test images...")
        reg.fit()
        pred = reg.predict(features_test)
        features_test["pred"] = pred
        features_test = features_test[[features_test.columns[-1]] + features_test.columns.tolist()[:-1]]
        features_test.to_csv("output/preds/{}-preds_{}.csv".format(os.path.basename(test_dir), label))

        print(features_test.pred.describe())
        print(reg.y.describe())

        return features_test

    return

def interpret(label, training_dirs, interpret_dir, file='jpg', n=None, ground_truth=True, save=False, **kwargs):
    reg = get_regressor(label, training_dirs)
    X = reg.X
    y = reg.y
    interpreter = Interpreter()
    interpreter.fit(X, y)

    features_interpret = FeatureExtractor(interpret_dir).get_features()
    if ground_truth:
        # fix this later - something wrong with loading if _aligned not used
        labels = LabelLoader(interpret_dir).get_labels(normalization=True)
        features_interpret = merge_x_y(features_interpret, labels).dropna(subset=[label])

    samples = features_interpret if n is None else features_interpret.sample(n)
    for i, sample in samples.iterrows():
        print(*sample[['Face name', 'Source']])
        sample_img = interpreter.get_img(
            *sample[['Face name', 'Source']],
            subd=os.path.basename(interpret_dir) if sample['Source'] != os.path.basename(interpret_dir) else None,
            d=os.path.dirname(interpret_dir),
            file=file
        )
        print(sample['Face name'])
        interpreter.explain_img(
            sample_img, 
            label=label,
            name=sample['Face name'],
            ground_truth = sample[label] if ground_truth else None,
            **kwargs
        )



def get_regressor(label, image_dirs):
    print("Generating {} regressor".format(label))
    filename = "models/regressor_{}.pkl".format(label)
    try:
        return pickle.load(open(filename, 'rb'))
    except (FileNotFoundError, EOFError, AttributeError):
        concats = []
        for image_dir in image_dirs:
            print("Extracting training features for {}...".format(image_dir))
            features_train = FeatureExtractor(image_dir).get_features()

            print("Extracting labels for {}...".format(image_dir))
            labels = LabelLoader(image_dir).get_labels()

            concats.append(merge_x_y(features_train, labels))

        df = pd.concat(concats, axis=0, join='inner', keys=[os.path.basename(image_dir) for image_dir in image_dirs])

        df.to_csv("output/train-data.csv")
        # print(df['Source'])
        reg = Regressor(df, label)

        # print("Fitting model...")
        # reg.fit(split=False)

        pickle.dump(reg, open(filename, 'wb'))

        return


def merge_x_y(X, y):
    return pd.merge(y, X, on="Face name", how="inner", validate="one_to_one")
