import pandas as pd

# Libraries for custom Transformer
from sklearn.base import BaseEstimator, TransformerMixin


# Helper functions
from utils.helper import pipelinetools_helper


class Debug(BaseEstimator, TransformerMixin):
    def __init__(self, debugMsg):
        self.debugMsg = debugMsg

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("------------------------")
        print(f"[Debug: {self.debugMsg}] X:\n", X)
        return X


### Date Cleaning Transformers ###
class AddColnamesTrans(BaseEstimator, TransformerMixin):
    def __init__(self, colnames):
        self.colnames = colnames

    def fit(self, X, y=None):
        print("[AddColnames]: fit()")
        return self

    def transform(self, X, y=None):
        print("[AddColnames]: transform()")
        return pd.DataFrame(X, columns=self.colnames)


class AstypeTrans(BaseEstimator, TransformerMixin):
    def __init__(self, col_type_dict={}):
        self.col_type_dict = col_type_dict

    def fit(self, X, y=None):
        print("[AstypeTrans]: fit()")
        return self

    def transform(self, X, y=None):
        ### Do not change this ###
        print("[AstypeTrans]: transform()")

        X_ = X.copy()
        if isinstance(X, pd.Series):  # Convert Series to Dataframe
            print("[AstypeTrans]: Converting pandas series to dataframe")
            X_ = X_.to_frame()
        ### Do not change this ###

        for key, val in self.col_type_dict.items():
            X_[key] = X_[key].astype(val)

        return X_


class ParseDateTrans(BaseEstimator, TransformerMixin):
    def __init__(self, date_colname):
        self.date_colname = date_colname

    def fit(self, X, y=None):
        print("[ParseDateTrans]: fit()")
        return self

    def transform(self, X, y=None):
        ### Do not change this ###
        print("[ParseDateTrans]: transform()")

        X_ = X.copy()
        if isinstance(X, pd.Series):  # Convert Series to Dataframe
            print("[ParseDateTrans]: Converting pandas series to dataframe")
            X_ = X_.to_frame()
        ### Do not change this ###

        X_["DateParsed"] = pd.to_datetime(X_[self.date_colname], format="%d/%m/%Y")
        X_["DateParsedYear"] = X_["DateParsed"].dt.year
        X_["DateParsedMonth"] = X_["DateParsed"].dt.month

        X_.drop(["Date", "DateParsed"], axis=1, inplace=True)
        return X_


### Date Analytics Transformers ###
class DataBinning(BaseEstimator, TransformerMixin):
    def __init__(self, bin_info=None):
        self.bin_info = bin_info

    def fit(self, X, y=None):
        print("[DataBinning]: fit()")
        if self.bin_info == None:
            print("[DataBinning]: auto binning...")
        return self

    def transform(self, X, y=None):
        print("[DataBinning]: transform()")
        X_ = X.copy()

        for colname, bins, labels, dtype in self.bin_info:
            X_[f"{colname}Bin"] = pd.cut(X_[colname], bins=bins, labels=labels).astype(
                dtype
            )
            X_.drop([colname], axis=1, inplace=True)

        return X_


class AddDropFeatBase(BaseEstimator, TransformerMixin):
    def __init__(self, add_colnames, del_colnames):
        self.add_colnames = add_colnames
        self.del_colnames = del_colnames

    def fit(self, X, y=None):
        return self

    def drop_cols(self, X):
        X.drop(self.del_colnames, axis=1, inplace=True)

    def transform(self, X, y=None):
        raise NotImplementedError("Please rewrite [AddDropFeatBase] transform method!")


class AddDropFeat(AddDropFeatBase):
    def __init__(self, add_colnames=None, del_colnames=None):
        AddDropFeatBase.__init__(self, add_colnames, del_colnames)
        self.add_colnames = add_colnames
        self.del_colnames = del_colnames

    def fit(self, X, y=None):
        print("[AddDropFeat]: fit()")
        return self

    def transform(self, X):
        print("[AddDropFeat]: transform()")
        X_ = X.copy()

        self.drop_cols(X_)
        return X_


### Encoder Transformers ###
class HighCardAggregation(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        print("[HighCardAggregation]: fit()")
        return self

    def transform(self, X, y=None):
        print("[HighCardAggregation]: transform()")
        X_ = X.copy()

        for col in list(X_.columns):
            X_[col], trans_list = pipelinetools_helper.cumulatively_categorise(
                column=X_[col], threshold=0.75
            )

        return X_
