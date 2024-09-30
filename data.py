import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch import Tensor

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from typing import Dict, List

import time
import gc

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch import Tensor

from sklearn.metrics import precision_recall_fscore_support, mean_squared_error, confusion_matrix, roc_auc_score
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold

from typing import Dict, List, Tuple, Union

import time
import gc


class DataWrapper:
    def __init__(self, data: Union[pd.DataFrame, pd.Series]):
        """
        Initialize the DataWrapper with a DataFrame or Series.

        Args:
            data (Union[pd.DataFrame, pd.Series]): The data to wrap.
        """
        self.data = data

    @property
    def names(self) -> List[str]:
        """
        Get the column names of the data.

        Returns:
            List[str]: List of column names.
        """
        return self.data.columns.values.tolist()

    @property
    def types(self) -> pd.DataFrame:
        """
        Get the data types of the columns in the data.

        Returns:
            pd.DataFrame: DataFrame with column names and their data types.
        """
        dtype = {idx: self.data[idx].dtype for idx in self.data.columns.values}
        ctr = pd.DataFrame([dtype]).T
        ctr = ctr.rename(columns={0: 'datatype'})
        ctr.index.names = ["Column"]
        return ctr


class DataMiner:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        X: Union[pd.DataFrame, pd.Series],
        y: Union[pd.DataFrame, pd.Series]
    ):
        """
        Initialize the DataMiner with a DataFrame, features, and target.

        Args:
            dataframe (pd.DataFrame): The main DataFrame.
            X (Union[pd.DataFrame, pd.Series]): The features.
            y (Union[pd.DataFrame, pd.Series]): The target.
        """
        if dataframe is None:
            raise ValueError("DataFrame cannot be null")
        self._check_df(dataframe)
        self.df = DataWrapper(dataframe)
        self.X = DataWrapper(X)
        self.y = y

    @staticmethod
    def _check_df(df: pd.DataFrame) -> None:
        """
        Check if the dataframe is valid.

        Args:
            df (pd.DataFrame): The DataFrame to check.

        Raises:
            AssertionError: If the DataFrame is empty or has duplicate columns.
        """
        assert df.shape[0] > 0 and df.shape[1] > 0, "DataFrame is Empty"
        assert len(df.columns.values) == len(set(df.columns.values)), "DataFrame has duplicate columns"

    def count_nulls(self) -> pd.DataFrame:
        """
        Count missing values per column, grouped by column name.

        Returns:
            pd.DataFrame: DataFrame with count and percentage of null values per column.
        """
        df_t = pd.DataFrame(self.df.data.isnull().sum()).rename(columns={0: "count"})
        df_t["percent_null"] = 100.0 * df_t["count"] / self.df.data.shape[0]
        df_t.index.names = ["Column"]
        return df_t.sort_values("percent_null", ascending=False)

    def count_distinct(self) -> pd.DataFrame:
        """
        Count distinct values per column.

        Returns:
            pd.DataFrame: DataFrame with count of distinct values per column.
        """
        unique_counts = {idx: self.df.data[idx].nunique() for idx in self.df.data.columns.values}
        unique_ctr = pd.DataFrame([unique_counts]).T
        unique_ctr = unique_ctr.rename(columns={0: 'count'})
        unique_ctr.index.names = ["Column"]
        return unique_ctr.sort_values("count", ascending=False)

    def concatenate(self) -> 'DataMiner':
        """
        Concatenate X and y into a single DataFrame.

        Returns:
            DataMiner: The updated DataMiner object.
        """
        if self.X is None or self.y is None:
            raise ValueError("X and y cannot be None")
        if self.X.data.shape[0] != self.y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if self.y.shape[1] < 1:
            raise ValueError("y must have at least one column")
        self.df = pd.concat([pd.DataFrame(self.X.data), pd.DataFrame(self.y[:, 0].tolist(), columns=['target'])], axis=1)
        return self

    def concatenate_with(self, other: Union[pd.DataFrame, np.ndarray], attr: str = 'df', axis: int =  0) -> 'DataMiner':
        """
        Concatenate another DataFrame or array with the existing data.

        Args:
            other (Union[pd.DataFrame, np.ndarray]): The data to concatenate.
            attr (str, optional): The attribute to concatenate with. Defaults to 'df'.
            axis (int, optional): The axis to concatenate along. Defaults to 0.

        Returns:
            DataMiner: The updated DataMiner object.
        """
        if attr == 'X':
            if not isinstance(other, (np.ndarray, pd.DataFrame)):
                raise ValueError("other must be a numpy array or pandas DataFrame")
            self.X = np.concatenate((self.X.data, other), axis=axis)
        elif attr == 'df':
            if not isinstance(other, pd.DataFrame):
                raise ValueError("other must be a pandas DataFrame")
            self.df = pd.concat([self.df.data, other], axis=axis)
        else:
            raise ValueError("attr must be either 'X' or 'df'")
        return self

    def filter_values(self, filter: Dict[str, List[float]], inplace: bool = True) -> Union[None, pd.DataFrame]:
        """
        Filter the data based on the provided filter.

        Args:
            filter (Dict[str, List[float]]): The filter to apply.
            inplace (bool, optional): Whether to modify the original data. Defaults to True.

        Returns:
            Union[None, pd.DataFrame]: The filtered data if inplace is False, otherwise None.
        """
        df_filtered = self.df.data
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

    def delete(self, columns: List[str], attr: str = 'df', inplace: bool = True) -> Union[None, pd.DataFrame]:
        """
        Delete columns from the data.

        Args:
            columns (List[str]): The columns to delete.
            attr (str, optional): The attribute to delete from. Defaults to 'df'.
            inplace (bool, optional): Whether to modify the original data. Defaults to True.

        Returns:
            Union[None, pd.DataFrame]: The updated data if inplace is False, otherwise None.
        """
        assert len(columns) > 0, "Column list passed for dropping is empty"
        match attr:
            case 'df':
                cur_cols = set(self.df.data.columns)
                drop_columns = list(set(columns).intersection(cur_cols))
                if inplace:
                    self.df = self.df.data.drop(drop_columns, axis=1, inplace=inplace)
                else:
                    return self.df.data.drop(drop_columns, axis=1, inplace=inplace)
            case 'X':
                cur_cols = set(self.X.data.columns)
                drop_columns = list(set(columns).intersection(cur_cols))
                if inplace:
                    self.X = self.X.data.drop(drop_columns, axis=1, inplace=inplace)
                else:
                    return self.X.data.drop(drop_columns, axis=1, inplace=inplace)

    def find_nans(self) -> List[str]:
        """
        Find columns with NaN values.

        Returns:
            List[str]: List of column names with NaN values.
        """
        columns = self.df.data.columns.values.tolist()
        cols = [colname for colname in columns if np.sum(pd.isnull(self.df.data[colname])) > 0]
        return list(np.sort(cols))

    def split(self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42, use_val: bool = False) -> Tuple:
        """
        Split the data into training, testing, and validation sets.

        Args:
            test_size (float, optional): The proportion of the data to include in the test set. Defaults to 0.2.
            val_size (float, optional): The proportion of the data to include in the validation set. Defaults to 0.1.
            random_state (int, optional): The seed used to shuffle the data before splitting. Defaults to 42.
            use_val (bool, optional): Whether to include a validation set. Defaults to False.

        Returns:
            Tuple: A tuple containing the training data, testing data, validation data, training labels, testing labels, and validation labels.
        """
        if self.X is None or self.y is None:
            self.split_df()
        X, y = shuffle(self.X.data, self.y)
        X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=test_size + val_size, random_state=random_state)
        if use_val:
            X_test, X_val, y_test, y_val = train_test_split(X_tmp, y_tmp, test_size=test_size / (test_size + val_size), random_state=random_state)
            return X_train, X_test, X_val, y_train, y_test, y_val
        else:
            return X_train, X_tmp, y_train, y_tmp

    def split_df(self, key: str = 'target') -> None:
        """
        Split the DataFrame into X and y.

        Args:
            key (str, optional): The column name to use as the target. Defaults to 'target'.
        """
        if key not in self.df.data.columns:
            raise ValueError("DataFrame must contain a column named key provided or 'target'")
        self.X = self.df.data.drop(key, axis=1)
        self.y = self.df.data[key]

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the root mean squared error.

        Args:
            y_true (np.ndarray): The true values.
            y_pred (np.ndarray): The predicted values.

        Returns:
            float: The root mean squared error.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return mean_squared_error(y_true, y_pred) ** 0.5

    @staticmethod
    def matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: List = None, sample_weight: List = None) -> pd.DataFrame:
        """
        Calculate the confusion matrix.

        Args:
            y_true (np.ndarray): The true values.
            y_pred (np.ndarray): The predicted values.
            labels (List, optional): The labels to use. Defaults to None.
            sample_weight (List, optional): The sample weights to use. Defaults to None.

        Returns:
            pd.DataFrame: The confusion matrix.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if labels is None:
            labels = np.unique(np.concatenate((y_true, y_pred)))
        matrix = confusion_matrix(y_true, y_pred, labels, sample_weight)
        matrix = pd.DataFrame(matrix, index=labels, columns=labels)

        matrix["Actual Counts"] = matrix.sum(axis=1)
        predicted_counts = pd.DataFrame(matrix.sum(axis=0)).T
        matrix = pd.concat([matrix, predicted_counts], ignore_index=True)

        new_index = list(labels)
        new_index.append("Predicted Counts")
        matrix.index = new_index
        matrix.index.names = ["Actual"]
        matrix.columns.names = ["Predicted"]

        actual_counts = matrix["Actual Counts"].values[:-1]
        predicted_counts = matrix[matrix.index == "Predicted Counts"].values[0][:-1]
        good_predictions = list()
        for label in labels:
            good_predictions.append(matrix[label].values[label])

        recall = 100 * np.array(good_predictions) / actual_counts
        precision = 100 * np.array(good_predictions) / predicted_counts
        recall = np.append(recall, [np.nan])
        matrix["Recall %"] = recall
        precision = pd.DataFrame(precision).T
        matrix = pd.concat([matrix, precision], ignore_index=True)
        new_index.append("Precision %")
        matrix.index = new_index
        matrix.index.names = ["Actual"]
        matrix.columns.names = ["Predicted"]
        matrix.fillna(-997, inplace=True)
        matrix = matrix.astype(int)
        matrix.replace(-997, np.nan, inplace=True)
        return matrix

    def xval(self, model: callable, fn: callable, cv: int = 4, proba: bool = True):
        """
        Perform cross-validation.

        Args:
            model (callable): The model to use.
            fn (callable): The function to use for evaluation.
            cv (int, optional): The number of folds. Defaults to 4.
            proba (bool, optional): Whether to use probabilities. Defaults to True.
        """
        X, y = shuffle(self.X.data, self.y)
        kf = KFold(n_splits=cv)
        results = {}
        i = 0
        for train_index, test_index in kf.split(X):
            X_train, y_train = X.iloc[train_index], y.iloc[train_index]
            X_test, y_test = X.iloc[test_index], y.iloc[test_index]
            model = model()
            start = time.time()
            print("Starting Processing for Group %s of %s." % ((i + 1), cv))
            model.fit(X_train, y_train)
            if proba:
                y_score = model.predict_proba(X_test)
            else:
                y_score = model.predict(X_test)
            res = fn(y_test, y_score, data=X_test)
            results[i] = res
            end = time.time()
            print("Group %s of %s done. Time taken = %.1f" % ((i + 1), cv, end - start))
            i += 1
            gc.collect()
        gc.collect()
        return results

    @staticmethod
    def plot_auc(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> None:
        """
        Plot the ROC AUC curve.

        Args:
            y_true (np.ndarray): The true values.
            y_pred (np.ndarray): The predicted values.
            threshold (float, optional): The threshold to use. Defaults to 0.5.
        """
        import seaborn as sns
        from matplotlib.colors import ListedColormap
        from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, f1_score, accuracy_score, confusion_matrix
        import matplotlib.pyplot as plt
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)

        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        aupr = average_precision_score(y_true, y_pred)

        plt.figure(figsize=(20, 4))

        # Plot ROC AUC curve
        plt.subplot(1, 3, 1)
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC AUC Curve')
        plt.legend(loc="lower right")

        # Plot Precision-Recall curve
        plt.subplot(1, 3, 2)
        plt.plot(recall, precision, label='AUPR = %0.2f' % aupr)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="upper right")

        y_pred = y_pred > threshold
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cm = np.array([[tp, fp], [fn, tn]])
        plt.subplot(1, 3, 3)
        # Define the colors for each cell
        colors = ["green", "red", "orange", "green"]
        cm_sb = np.array([[1.0, 2.0], [3.0, 4.0]])
        # Create the ListedColormap
        cmap = ListedColormap(colors)
        sns.heatmap(cm_sb, annot=False, cmap=cmap, fmt='g', cbar=False, annot_kws={"size": 32})
        plt.xlabel('Actual', labelpad=-10)
        plt.ylabel('Predicted')
        plt.xticks([0.5, 1.5], ['Positive', 'Negative'])
        plt.yticks([0.5, 1.5], ['Positive', 'Negative'])
        names = [["TP", "FP"], ["FN", "TN"]]
        # Add value annotations to the heatmap
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.annotate(names[i][j] + " = " + str(cm[i][j]), xy=(j + 0.5, i + 0.5), ha="center", va="center")

        plt.title('Confusion Matrix')
        plt.show()

        accuracy = accuracy_score(y_true, y_pred)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))

        roc_auc = roc_auc_score(y_true, y_pred)
        print("ROC AUC: %.2f" % roc_auc)

        aupr = average_precision_score(y_true, y_pred)
        print("AUPR: %.2f" % aupr)

        f1 = f1_score(y_true, y_pred)
        print("F1: %.2f" % f1)

    def plot_corr(self, threshold: float = 0, figsize: Tuple = (6, 5), fmt: str = '.2f', spearman: bool = False):
        """
        Plot the correlation matrix.

        Args:
            threshold (float, optional): The threshold to use. Defaults to 0.
            figsize (Tuple, optional): The figure size. Defaults to (6, 5).
            fmt (str, optional): The format to use. Defaults to '.2f'.
            spearman (bool, optional): Whether to use Spearman correlation. Defaults to False.
        """
        import seaborn as sns
        import matplotlib.pyplot as plt

        corr = self.df.data.corr()
        if spearman:
            from scipy import stats
            res = stats.spearmanr(self.df.data.values)
            corr = pd.DataFrame(res.statistic, index=corr.index, columns=corr.columns)

        corr = corr.where(np.abs(corr) > threshold, 0)

        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=figsize)

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(240, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, cbar_kws={"shrink": .8}, center=0,
                    square=True, linewidths=.5, annot=True, fmt=fmt)
        plt.title("Column Correlation Heatmap")
        plt.show()

        return corr

    def plot_scatter(self, labels: List, figsize: Tuple = (12, 12)):
        """
        Plot a scatter plot.

        Args:
            labels (List): The labels to use.
            figsize (Tuple, optional): The figure size. Defaults to (12, 12).
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(self.X.data, self.y)
        for i, txt in enumerate(labels):
            ax.annotate(txt, (self.X.data[i], self.y[i]))
        plt.show()

    @staticmethod
    def plot_pr(y_true: np.ndarray, y_pred: np.ndarray):
        from inspect import signature
        from sklearn.metrics import precision_recall_curve, average_precision_score
        import matplotlib.pyplot as plt

        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        step_kwargs = ({'step': 'post'}
                    if 'step' in signature(plt.fill_between).parameters
                    else {})
        plt.figure(figsize=(8, 8))
        plt.step(recall, precision, color='b', alpha=0.2,
                where='post')
        ap = average_precision_score(y_true, y_pred)
        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall curve: Average Precision = %.4f' % (ap));
        plt.show()
