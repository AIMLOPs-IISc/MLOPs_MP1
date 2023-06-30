from typing import List, Union
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder 
#import datetime as dt

from sklearn.base import BaseEstimator, TransformerMixin

class WeekdayImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weekday' column by extracting dayname from 'dteday' column """

    def __init__(self, variables:str, ref_variables:str, len_day_name:int):
        if not isinstance(variables, str):
            raise ValueError("variables should be a str")
        if not isinstance(ref_variables, str):
            raise ValueError("reference variables should be a str")
        if not isinstance(len_day_name, int):
            raise ValueError("len day name should be an int")

        self.variables = variables
        self.ref_variables = ref_variables
        self.len_day_name = len_day_name
        
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X1 = X.copy()
        self.wkday_null_idx = X1[X1[self.variables].isnull() == True].index
        fit_fn = lambda x: x.dt.day_name().apply(lambda a: a[:self.len_day_name])
        X1.loc[self.wkday_null_idx, self.variables] = fit_fn(X1.loc[self.wkday_null_idx, self.ref_variables])
        return X1
     
class WeathersitImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """

    def __init__(self, variables:str):
        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.fill_value=X[self.variables].mode()[0]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X1 = X.copy()
        X1[self.variables]=X1[self.variables].fillna(self.fill_value)
        return X1
    

class Mapper(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable mapper:
    Treat column as Ordinal categorical variable, and assign values accordingly
    """

    def __init__(self, variables: str, mappings: Union[dict, None] = None):

        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

        self.variables = variables
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        # create Mappings if not provided
        if self.mappings is None:
            data = X[self.variables].value_counts(ascending=True).index
            print(self.mappings)
            self.mappings = {val:cnt for cnt,val in enumerate(data)}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        #for feature in self.variables:
        X[self.variables] = X[self.variables].map(self.mappings).astype(int)
        #X[self.variables] = X[self.variables].map(self.mappings)
        return X
        
        
class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values: 
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self, variables: str, lower_bound_val:str, upper_bound_val:str):
        if not isinstance(variables, str):
            raise ValueError("variables should be a str")
        if not isinstance(lower_bound_val, str):
            raise ValueError("lower bound variables should be a str")
        if not isinstance(upper_bound_val, str):
            raise ValueError("upper bound variables should be a str")

        self.variables = variables
        self.lower_bound_val = lower_bound_val
        self.upper_bound_val = upper_bound_val

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.q1 = X.describe()[self.variables][self.lower_bound_val]
        self.q3 = X.describe()[self.variables][self.upper_bound_val]
        self.iqr = self.q3 - self.q1
        self.lower_bound = self.q1 - (1.5 * self.iqr)
        self.upper_bound = self.q3 + (1.5 * self.iqr)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X=X.copy()
        for i in X.index:
            if X.loc[i,self.variables] > self.upper_bound:
                X.loc[i,self.variables]= self.upper_bound
            if X.loc[i,self.variables] < self.lower_bound:
                X.loc[i,self.variables]= self.lower_bound
        return X
    
class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode weekday column """

    def __init__(self, variables:str):
        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.encoder = OneHotEncoder(sparse_output=False)
        self.encoder.fit(X[[self.variables]])
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X1=X.copy().reset_index(drop=True)
        temp=self.encoder.transform(X1[[self.variables]])
        x1_new = pd.DataFrame(temp, columns=self.encoder.get_feature_names_out())
        # print(len(X1), X1.columns, len(x1_new), x1_new.columns)
        # print(X1.head(2))
        # print(x1_new.head(2))
        X1 = pd.concat([X1, x1_new], axis=1)
        #print(len(X1), X1.columns)
        return X1
        
