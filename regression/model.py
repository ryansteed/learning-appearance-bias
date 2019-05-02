from regression.utils import Ticker
from regression.feature_extraction import LabelLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.base import clone
import seaborn as sns
# from scipy.stats import ttest_ind


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
        reg = clone(self.reg)
        n = 10
        kf = KFold(n_splits=n)

        preds = []
        ticker = Ticker(n, "Validated", "folds", verbose=True)
        for train, test in kf.split(self.X):
            X_train, X_test = self.X.iloc[train], self.X.iloc[test]
            y_train, y_test = self.y.iloc[train], self.y.iloc[test]
            reg.fit(X_train, y_train)
            preds += zip(reg.predict(X_test), y_test)
            ticker.update()
        pred, actual = zip(*preds)
        sns.scatterplot(x=actual, y=pred).get_figure().savefig("output/scatter.png")

    def predict(self, test_df):
        X = self.make_X(test_df)
        return self.reg.predict(X)

    def make_X(self, df):
        to_drop = [x for x in LabelLoader.base_labels + ['Face name', 'Source'] if x in df.columns]
        return df.drop(columns=to_drop, axis=1)

    def make_X_y(self, df):
        return self.make_X(df), df[self.label]
