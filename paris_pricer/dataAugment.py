from turtle import distance
import pandas as pd 
import numpy as np
import os
import ast
import geopandas as gpd
import math
from tqdm import tqdm

# 

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
        """Load external data for Ile-de-France
            
        """
        
        self.df = df

        #communes & arrondisements
        self.idf_reg = gpd.read_file('communes-dile-de-france-au-01-janvier.shp', ignore_geometry=True).rename(columns={'insee': 'l_codinsee'})
        self.idf_reg['nomcom'] = self.idf_reg['nomcom'].str.encode('ISO-8859-1').str.decode('utf-8')
        
        #train stations
        self.gares = pd.read_json('emplacement-des-gares-idf.json')
        self.gares[['longitude', 'latitude']] = self.gares.geo_point_2d.apply(pd.Series)
    
    def add_subdivisions(self) -> pd.DataFrame:
        """Adds communes/arrondisements to locations
        
        Returns:
            pd.merge: merged original dataframe with ile-de-france geometric data
        """
        self.df['l_codinsee'] = self.df['l_codinsee'].astype(float)
        return pd.merge(self.df, self.idf_reg, on='l_codinsee')
       
    def count_public_transport_spots(self,
                                     distance_to_station: int = 200, 
                                     subdivisons: int = 150) -> pd.DataFrame:
        """
        Count the number of public transport spots within 200 meters of each point in df1.
        
        Args:
            subdivisons: split the df into x subdivisions (for memory purposes)
            
        Returns:
            self.df: returns df with transport nearby
        """
        coords2 = np.radians(self.gares[['latitude', 'longitude']].to_numpy())
        lats2, lons2 = coords2[:, 0], coords2[:, 1]
        
        sub_dfs = np.array_split(self.df, subdivisons)

        spots = []
        for i in tqdm(range(subdivisons)):
            sub_df = sub_dfs[i]
            sub_coords1 = np.radians(sub_df[['latitude', 'longitude']].to_numpy())
            lats1, lons1 = sub_coords1[:, 0], sub_coords1[:, 1]


            a = np.sin((lats2 - lats1[:, None])/2)**2 + \
                np.cos(lats1[:, None]) * np.cos(lats2) * \
                np.sin((lons2 - lons1[:, None])/2)**2

            c = 2 * np.arcsin(np.sqrt(a))
            distances = c * 6371 * 1000

            transport_spots = np.sum(distances <= distance_to_station, axis=1)
            spots.extend(transport_spots)

            
        self.df['public_transport_spots'] = spots
        return self.df
