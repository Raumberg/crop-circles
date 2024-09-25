import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch import Tensor

from typing import Dict, List

class DataWrapper:
    def __init__(self, data: pd.DataFrame | pd.Series):
        self.data = data

    @property
    def names(self) -> list:
        return self.data.columns.values.tolist()
    
    
    @property
    def types(self) -> pd.DataFrame:
        dtype = {}
        for idx in self.data.columns.values:
            dt = self.data[idx].dtype
            dtype[idx]=dt
        ctr = pd.DataFrame([dtype]).T
        ctr = ctr.rename(columns={0: 'datatype'})
        ctr.index.names = ["Column"]
        return ctr


class DataMiner:
    def __init__(
            self,
            dataframe: pd.DataFrame,
            X: pd.DataFrame | pd.Series,
            y: pd.DataFrame | pd.Series 
        ):
        if dataframe is None:
            raise ValueError("DataFrame cannot be null")
        self._check_df(dataframe)
        self.df = DataWrapper(dataframe)
        self.X = DataWrapper(X)
        self.y = y

    @staticmethod
    def _check_df(df: pd.DataFrame) -> None:
        """
        Check if the dataframe is valid
        """
        assert df.shape[0] > 0 and df.shape[1] > 0 , "DataFrame is Empty"
        assert len(df.columns.values) == len(set(df.columns.values)) , "DataFrame has duplicate columns"

    def count_nulls(self) -> pd.DataFrame:
        """
        Count missing values per column, grouped by column name
        """
        df_t = pd.DataFrame(self.df.isnull().sum()).rename(columns={0:"count"})
        df_t["percent_null"] = 100.0 * df_t["count"] / self.df.shape[0]
        df_t.index.names = ["Column"]
        return df_t.sort_values("percent_null", ascending=False)
    
    def count_distinct(self) -> pd.DataFrame:
        unique_counts = {}
        for idx in self.df.columns.values:
            #cnt=len(df[idx].unique())
            cnt = self.df[idx].nunique()
            unique_counts[idx]=cnt
        unique_ctr = pd.DataFrame([unique_counts]).T
        unique_ctr = unique_ctr.rename(columns={0: 'count'})
        unique_ctr.index.names = ["Column"]
        return unique_ctr.sort_values("count",ascending=False)
    
    def concatenate(self) -> None:
        if self.X is None or self.y is None:
            raise ValueError("X and y cannot be None")
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if self.y.shape[1] < 1:
            raise ValueError("y must have at least one column")
        self.df = pd.concat([pd.DataFrame(self.X), pd.DataFrame(self.y[:,0].tolist(),columns=['target'])], axis=1)
        return self
    
    def concatenate_with(self, other: pd.DataFrame | np.ndarray, attr: str = 'df', axis=0) -> None:
        if attr == 'X':
            if not isinstance(other, (np.ndarray, pd.DataFrame)):
                raise ValueError("other must be a numpy array or pandas DataFrame")
            self.X = np.concatenate((self.X, other), axis=axis)
        elif attr == 'df':
            if not isinstance(other, pd.DataFrame):
                raise ValueError("other must be a pandas DataFrame")
            self.df = pd.concat([self.df, other], axis=axis)
        else:
            raise ValueError("attr must be either 'X' or 'df'")
        return self

    def filter_values(self, filter: Dict[str, list], inplace: bool = True) -> None | pd.DataFrame:
        df_filtered = self.df
        for feature in filter:
            values = filter[feature]
            if len(values) == 1:
                df_filtered = df_filtered[df_filtered[feature] <= values[0]]
            elif len(values) == 2:
                df_filtered = df_filtered[(df_filtered[feature] >= values[0]) & (df_filtered[feature] <= values[1])]
        if inplace:
            self.df = df_filtered
        else:
            return df_filtered
        
    def delete(self, columns: list, attr: str = 'df', inplace: bool = True) -> None | pd.DataFrame:
        assert len(columns) > 0, "Column list passed for dropping is empty"
        match attr:
            case 'df':
                cur_cols = set(self.df.columns)
                drop_columns = list(set(columns).intersection(cur_cols))
                if inplace:
                    self.df.drop(drop_columns, axis=1, inplace=inplace)
                else:
                    return self.df.drop(drop_columns, axis=1, inplace=inplace)
            case 'X':
                cur_cols = set(self.X.columns)
                drop_columns = list(set(columns).intersection(cur_cols))
                if inplace:
                    self.X.drop(drop_columns, axis=1, inplace=inplace)
                else:
                    return self.X.drop(drop_columns, axis=1, inplace=inplace)
                
    def find_nans(self) -> list:
        columns = self.df.columns.values.tolist()
        cols = list()
        for colname in columns:
            if(np.sum(pd.isnull(self.df[colname])) > 0):
                cols.append(colname)
        return list(np.sort(cols))
    
    def split(self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42, use_val: bool = False) -> tuple:
        """
        Split the data into training, testing, and validation sets.

        Args:
        test_size (float, optional): The proportion of the data to include in the test set. Defaults to 0.2.
        val_size (float, optional): The proportion of the data to include in the validation set. Defaults to 0.1.
        random_state (int, optional): The seed used to shuffle the data before splitting. Defaults to 42.

        Returns:
        tuple: A tuple containing the training data, testing data, validation data, training labels, testing labels, and validation labels.
        """
        if self.X is None or self.y is None:
            self.split_df()
        X = self.X
        y = self.y
        X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=test_size + val_size, random_state=random_state)
        if use_val:
            X_test, X_val, y_test, y_val = train_test_split(X_tmp, y_tmp, test_size=test_size / (test_size + val_size), random_state=random_state)
            return X_train, X_test, X_val, y_train, y_test, y_val
        else:
            return X_train, X_tmp, y_train, y_tmp
    
    def split_df(self, key: str = 'target') -> None:
        """
        Split self.df into X and y, where y is the column named by key, or is defaulted to 'target'.
        """
        if key not in self.df.columns:
            raise ValueError("DataFrane must contain a column named key provided or 'target'")
        self.X = self.df.drop(key, axis=1)
        self.y = self.df[key]
