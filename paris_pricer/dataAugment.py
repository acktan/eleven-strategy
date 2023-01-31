import pandas as pd 
import numpy as np
import os
import ast
import geopandas as gpd
from shapely.geometry import Point


os.chdir(r"C:\Users\ckunt\OneDrive\Documents\Masters work\HEC\18. Eleven Strategy\eleven-strategy\data")

csv_files = ['mutations_d75_train_localized.csv',
             'mutations_d77_train_localized.csv',
             'mutations_d78_train_localized.csv',
             'mutations_d91_train_localized.csv',
             'mutations_d92_train_localized.csv',
             'mutations_d93_train_localized.csv',
             'mutations_d94_train_localized.csv',
             'mutations_d95_train_localized.csv',
             ]

file_path = 'data/'

class DataLoader():
    """Load, clean and combine mutuation csv files
    
    Args:
        csv_files = list of names of csv files
    
    """
    def __init__(self, csv_files: list[str]) -> None:
        self.csv_files = csv_files
    
    def combine_clean_files(self, 
                            explod: list[str] = 'l_codinsee') -> pd.DataFrame:
        """Combine csv files
        
        Returns:
            df: dataframe with combined dataframes
        """
        
        print('Getting dfs...')
        list_dfs = [pd.read_csv(f) for f in self.csv_files]
        
        print('Combining dfs...')
        df = pd.concat(list_dfs, ignore_index=True)
        
        print('Cleaning dfs...')
        l_cols = [col for col in df.columns if col.startswith('l_')]
        df[l_cols] = df[l_cols].applymap(ast.literal_eval)
        df = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
        
        print('Exploding columns...')
        return df.explode(explod)
    

class DataAugmentation(DataLoader):
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        
    def load_idf_data(self) -> None:
        """Load geometric data for Ile-de-France
            
        """
        
        self.idf_reg = gpd.read_file('communes-dile-de-france-au-01-janvier.shp', ignore_geometry=True).rename(columns={'insee': 'l_codinsee'})
        self.idf_reg['nomcom'] = self.idf_reg['nomcom'].str.encode('ISO-8859-1').str.decode('utf-8')
    
    def add_subdivisions(self) -> pd.DataFrame:
        """Adds communes/arrondisements to locations
        
        Returns:
            pd.merge: merged original dataframe with ile-de-france geometric data
        """
        self.df['l_codinsee'] = self.df['l_codinsee'].astype(float)
        return pd.merge(self.df, self.idf_reg, on='l_codinsee')


dl = DataLoader(csv_files)
a = dl.combine_clean_files()
dataAug  = DataAugmentation(a)
dataAug.load_idf_data()
merged_df = dataAug.add_subdivisions()



