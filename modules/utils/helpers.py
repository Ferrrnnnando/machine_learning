import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

# Data scaling
from sklearn.preprocessing import StandardScaler

# Model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

# Evaluation
from sklearn.metrics import mean_squared_error

# Libraries for custom Transformer
from sklearn.base import BaseEstimator, TransformerMixin

# Others
from collections import Counter

def printHello():
    print('Hello! I am helper modules')

def save_fig(folder_path, fig_id, tight_layout=True, fig_extension="png", resolution=300):
    # os.makedirs(folder_path, exist_ok=True)

    path = os.path.join(folder_path, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def extract_substring(s, substring):
    for t in substring:
        if t in s:
            return t
    return np.nan


# # High Cardinality Transformer: perform one-hot transformer for high-cardinality
# # columns while avoiding generating too much dummpy features. It will only one-hot
# # the classes with high proportion. A threshold can be set.

# def cumulatively_categorise(column,threshold=0.75,return_categories_list=True):
#     #Find the threshold value using the percentage and number of instances in the column
#     threshold_value=int(threshold*len(column))
#     #Initialise an empty list for our new minimised categories
#     categories_list=[]
#     #Initialise a variable to calculate the sum of frequencies
#     s=0
#     #Create a counter dictionary of the form unique_value: frequency
#     counts=Counter(column)

#     #Loop through the category name and its corresponding frequency after sorting the categories by descending order of frequency
#     for i,j in counts.most_common():
#         #Add the frequency to the global sum
#         s+=dict(counts)[i]
#         #Append the category name to the list
#         categories_list.append(i)
#         #Check if the global sum has reached the threshold value, if so break the loop
#         if s>=threshold_value:
#             break
#     #Append the category Other to the list
#     categories_list.append('Other')

#     #Replace all instances not in our new categories by Other  
#     new_column=column.apply(lambda x: x if x in categories_list else 'Other')

#     #Return transformed column and unique values if return_categories=True
#     if(return_categories_list):
#         return new_column,categories_list
#     #Return only the transformed column if return_categories=False
#     else:
#         return new_column


# # high_card_cols = ['Regionname', 'CouncilArea']
# class HighCardAggregation(BaseEstimator, TransformerMixin):
#     def __init__(self, high_card_cols):
#         self.cols = high_card_cols

#     def fit(self, X, y = None):
#         return self

#     def transform(self, X, y = None):
#         # X is a numpy array, the columns to with high cardinality
#         X_ = X.copy()
#         X_df = pd.DataFrame(X, columns=self.cols)
        
#         for col in self.cols:
#             # transformed_columns is a pandas dataframe
#             transformed_column, trans_list = cumulatively_categorise(column=X_df[col], threshold=0.75)
# #             print('transformed_column: \n', transformed_column.value_counts())
#             X_ = np.c_[X_, transformed_column.to_numpy()]
            
#         X_ = np.delete(X_, [i for i in range(len(self.cols))], 1)
#         return X_



# class FullPipeline:
#     def __init__(self, 
#         add_feat_pipe, 
#         del_feat_pipe, 
#         num_cols, 
#         cat_cols, 
#         feat_num_cols, 
#         feat_cat_cols, 
#         low_card_cols, 
#         high_card_cols,
#         ):

#         self.AddFeature = add_feat_pipe
#         self.DelFeature = del_feat_pipe
        
#         self.num_cols = num_cols
#         self.cat_cols = cat_cols
#         self.feat_num_cols = feat_num_cols
#         self.feat_cat_cols = feat_cat_cols
#         self.low_card_cols = low_card_cols
#         self.high_card_cols = high_card_cols


#         # Low cardinality columns
#         self.low_card_pipe = Pipeline(steps=[
#             ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
#         ])

#         # High cardinality columns
#         self.high_card_pipe = Pipeline(steps=[
#             ('aggregation', HighCardAggregation()),
#             ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
#         ])

#         # Categorical transformers
#         self.cat_trans = ColumnTransformer(transformers=[
#             ('low_card_pipeline', self.low_card_pipe, self.low_card_cols),
#             ('high_card_pipeline', self.high_card_pipe, self.high_card_cols)
#         ])

#         # Numerical transformer
#         self.num_trans = Pipeline(steps=[
#             # ('imputer', SimpleImputer(strategy='median')),
#             ('std_scaler', StandardScaler())
#         ])

#         # Numerical & Categorical column transformer
#         self.num_cat_trans = ColumnTransformer(transformers=[
#             ('num', self.num_trans, self.feat_num_cols),
#             ('cat', self.cat_trans, self.feat_cat_cols)
#         ])


#         self.feat_eng = Pipeline([
#             ('add_feat', self.AddFeature(self.num_cols + self.cat_cols)),
#             # ('delete_feat', DelFeature())     
#         ])

#         self.imputer = ColumnTransformer([
#             ('imputer_num', SimpleImputer(strategy='median'), self.num_cols),
#             ('imputer_cat', SimpleImputer(strategy='most_frequent'), self.cat_cols)
#         ])

#         self.clean_pipe = Pipeline([
#             ('imputer', self.imputer)
#         ])

#         self.preprocessor = Pipeline([
#             ('clean', self.clean_pipe),
#             ('feature_eng', self.feat_eng),
#             ('num_cat', self.num_cat_trans)
#         ])
    