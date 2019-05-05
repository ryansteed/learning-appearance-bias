from regression.utils import Ticker
from regression.feature_extraction import LabelLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.base import clone
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib
import matplotlib.pyplot as plt
import pickle
# from scipy.stats import ttest_ind


class Regressor:
    def __init__(self, df, label):
        self.df = df[df[label].notna()]
        self.label = label
        self.X, self.y = self.make_X_y(self.df)
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

    def cross_validate(self, cache=True):
        reg = clone(self.reg)
        n = 10
        kf = KFold(n_splits=n, shuffle=True, random_state=42)
        try:
            if cache:
                preds = pickle.load(open(self.make_preds_filename(), 'rb'))
            else:
                raise ValueError
        except (FileNotFoundError, ValueError):
            preds = []
            ticker = Ticker(n, "Validated", "folds", verbose=True)
            for i, split in enumerate(kf.split(self.X)):
                train, test = split
                X_train, X_test = self.X.iloc[train], self.X.iloc[test]
                y_train, y_test = self.y.iloc[train], self.y.iloc[test]
                reg.fit(X_train, y_train)
                preds += zip(reg.predict(X_test), y_test, np.repeat(i, y_test.shape[0]))
                ticker.update()
            pickle.dump(preds, open(self.make_preds_filename(), 'wb'))
        preds_df = pd.DataFrame(preds, columns=["pred", "actual", "fold"])
        self.df = self.df.merge(preds_df, how='left', left_on=self.df[self.label], right_on=preds_df.actual)

        self.df["Source_formatted"] = self.df["Source"].apply(self.format_source)
        self.df[["Source_formatted", "Face name", "actual", "pred", "fold"]].to_csv("output/preds.csv")

    def make_preds_filename(self):
        return "models/preds_{}.pkl".format(self.label)

    def chart(self, name, annotate=False, hue="Source_formatted"):
        print(self.reg)
        err = np.absolute(self.df[self.label] - self.df.pred)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax = sns.scatterplot(ax=ax, x=self.label, y="pred", hue=hue, data=self.df)
        plt.title("Regression Chart")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        if annotate:
            for i, txt in enumerate(self.df["Face name"]):
                if err.iloc[i] > err.quantile(0.9):
                    ax.annotate(txt.split("_")[0], (self.df[self.label].iloc[i], self.df.pred.iloc[i]))
        ax.annotate("Pearson Coefficient\n rho={0:.4f} p={1:.4f}".format(*pearsonr(self.df[self.label], self.df.pred)), (150, -250))
        ax.get_figure().savefig("output/{}.png".format(name), dpi=300)
        # plt.show()

    @staticmethod
    def format_source(src):
        translation = {
            "bmp": "Maximally Distinct Faces"
        }
        return translation.get(src, "300 Random Faces")

    def predict(self, test_df):
        X = self.make_X(test_df)
        return self.reg.predict(X)

    def make_X(self, df):
        to_drop = [x for x in LabelLoader.base_labels + ['Face name', 'Source'] if x in df.columns]
        return df.drop(columns=to_drop, axis=1)

    def make_X_y(self, df):
        return self.make_X(df), df[self.label]