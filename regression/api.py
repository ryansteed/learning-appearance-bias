from regression.feature_extraction import FeatureExtractor, LabelLoader
from regression.model import Regressor
import os
import pickle
import pandas as pd


def regress(label, **kwargs):
    if label is None:
        return regress_all(**kwargs)
    return regress_single(label, **kwargs)


def regress_all(**kwargs):
    for label in ["Attractive", "Competent", "Dominant", "Extroverted", "Likeable", "Trustworthy"]:
        regress_single(label, **kwargs)


def regress_single(label, image_dirs, test_dir=None, cross_validate=False):
    reg = get_regressor(label, image_dirs)

    if cross_validate:
        print("Cross validating...")
        reg.cross_validate()
        reg.chart("scatter", annotate=False)
        reg.chart("scatter_folds", annotate=False, hue="fold")
        reg.chart("scatter_annotated", annotate=True)

    if test_dir is not None:
        print("Extracting test features...")
        features_test = FeatureExtractor(
            test_dir
        ).get_features()

        print("Predicting test images...")
        pred = reg.predict(features_test)
        features_test["pred"] = pred
        features_test = features_test[[features_test.columns[-1]] + features_test.columns.tolist()[:-1]]
        features_test.to_csv("output/pred_{}.csv".format(label))

        print(features_test.pred.describe())
        print(reg.y.describe())

        # print(features_test[["Source", "pred"]])
        # print(features_test.groupby("Source")["pred"].mean())
        # print(features_test.groupby("Source")["pred"].std())
        # cats = np.unique(features_test["Source"])
        # print(cats)
        # print(ttest_ind(
        #     features_test[features_test["Source"] == cats[0]].pred,
        #     features_test[features_test["Source"] == cats[1]].pred
        # ))

        return features_test

    return


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
        reg = Regressor(df, label)

        # print("Fitting model...")
        # reg.fit(split=False)

        pickle.dump(reg, open(filename, 'wb'))

        return reg