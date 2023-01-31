from __future__ import annotations
import pandas as pd

import os
from typing import Union
import warnings

warnings.simplefilter('ignore', FutureWarning)


class Data:
    @staticmethod
    def load_data(path: str = None) -> dict[str, pd.DataFrame]:
        """This function loads the csv files Eleven provided us with."""
        if path is None:
            for root, folders, files in os.walk('.'):
                if 'mutations_d75_train_localized.csv' in files:
                    path = root
                    break
        data = {}
        for file in os.listdir(path):
            # We are only interested in csv files
            if not file.endswith('.csv'):
                continue
            data[file.split('.', 1)[0]] = pd.read_csv(f'{path}/{file}')
            # Dropping 'codservch' and 'refdoc' since they only contain NaNs along with 'Unnamed: 0.1' and 'Unnamed: 0'
            # that are useless
            data[file.split('.', 1)[0]] = data[file.split('.', 1)[0]].drop(columns=['codservch', 'refdoc',
                                                                                    'Unnamed: 0.1', 'Unnamed: 0'])
            # Filling the NaN values in 'valeurfonc' with 0
            data[file.split('.', 1)[0]] = data[file.split('.', 1)[0]].fillna(value={'valeurfonc': 0})

        return data

    @staticmethod
    def infer_dtypes(df: pd.DataFrame) -> dict[Union[str, int], str]:
        """This function infers the data types of each column"""
        df = df.copy()
        # Define the lower and upper thresholds for each data type
        t_lower = [-128, 0, -32_768, 0, -2_147_483_648, 0, -9_223_372_036_854_775_808, 0]
        t_upper = [127, 255, 32_767, 65_535, 2_147_483_647, 4_294_967_295, 9_223_372_036_854_775_807,
                   18_446_744_073_709_551_615]
        t_name = ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
        dtypes = dict()
        # The string columns remain the same
        for col in df.select_dtypes(include=[object]):
            try:
                df[col].astype('datetime64')
                dtypes[col] = 'datetime64'
            except ValueError:
                dtypes[col] = 'object'
        # The datetime columns remain the same
        for col in df.select_dtypes(include=['datetime64']):
            dtypes[col] = 'datetime64'
        # The boolean columns remain the same
        for col in df.select_dtypes(include=[bool]):
            dtypes[col] = bool
        # The numeric type depends on the values
        for col in df.select_dtypes(exclude=[object, 'datetime64', bool]):
            try:
                # Check if a column only contains integers
                if ((df[col] - df[col].astype(int)).sum() == 0) & (df[col].isna().sum() == 0):
                    # Get the lower bound index
                    l_possible = [bound for bound in t_lower if (bound - df[col].min()) <= 0]
                    l_index = t_lower.index(l_possible[0])
                    (df[col].max(), abs(df[col].min()))
                    # Get the upper bound index
                    u_possible = [bound for bound in t_upper if bound - df[col].max() >= 0]
                    u_index = t_upper.index(u_possible[0])
                    # Take the higher value between the lower and upper bounds index
                    dtypes[col] = t_name[max(l_index, u_index)]
                # For floating point numbers we always use float64 to ensure good precision
                else:
                    dtypes[col] = 'float64'
            except pd.errors.IntCastingNaNError:
                warnings.warn(f"{col} might be an integer if the wasn't for the NA values")
                dtypes[col] = 'float64'
        return dtypes

    @staticmethod
    def eval_columns(df: pd.DataFrame, columns: list[Union[str, int]] = None) -> pd.DataFrame:
        df = df.copy()
        if columns is None:
            columns = ['l_codinsee', 'l_section', 'l_idpar', 'l_idparmut', 'l_idlocmut']
        # Turning the columns from string to lists (applying it all at once does not work because of the NaNs)
        for col in columns:
            df.loc[:, col] = df.loc[:, col].apply(eval)
        return df

    @staticmethod
    def explode_df(df: pd.DataFrame, columns: list[Union[str, int]] = None) -> pd.DataFrame:
        """This function explodes (=expands) the DataFrame based on the provided columns."""
        df = df.copy()
        if columns is None:
            # We use 'l_codinsee' and 'l_section' as default values since the other have too many unique values and are
            # therefore not useful
            columns = ['l_codinsee', 'l_section']

        # Exploding the columns individually for the same reason as above
        for col in columns:
            df = df.explode(column=col)
        return df

    @classmethod
    def load_df(cls, path: str = None, explode: bool = True, drop_useless: bool = True) -> pd.DataFrame:
        """This is the main data loading method"""
        data = cls.load_data(path=path)
        df = pd.concat([data[key] for key in data])
        df = df.astype(dtype=cls.infer_dtypes(df))
        if explode:
            df = cls.eval_columns(df)
            df = cls.explode_df(df)
        if drop_useless:
            # These columns are categorical and have mostly unique values
            useless_columns = ['idmutation', 'idmutinvar', 'idopendata', 'l_idpar', 'l_idparmut', 'l_idlocmut']
            df = df.loc[:, set(df.columns).difference(useless_columns)]
        return df
