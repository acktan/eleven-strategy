import pandas as pd 
import numpy as np
import ast
import geopandas as gpd
from tqdm import tqdm
from shapely.geometry import Point

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
    def __init__(self, 
                 df: pd.DataFrame,
                 file_path: str) -> None:
        """Load external data for Ile-de-France
            
        """
        
        
        self.df = df
        self.file_path = file_path

        #communes & arrondisements
        self.idf_reg = gpd.read_file(self.file_path + 'communes-dile-de-france-au-01-janvier.shp', ignore_geometry=True).rename(columns={'insee': 'l_codinsee'})
        self.idf_reg['nomcom'] = self.idf_reg['nomcom'].str.encode('ISO-8859-1').str.decode('utf-8')
        
        #train stations
        self.gares = pd.read_json(self.file_path + 'emplacement-des-gares-idf.json')
        self.gares[['longitude', 'latitude']] = self.gares.geo_point_2d.apply(pd.Series)
        
        #crimerate
        self.df_crimerate2018 = pd.read_csv(self.file_path + 'crimerate2018.csv',sep=";")
        self.df_crimerate2018['coddep'] = self.df_crimerate2018['coddep'].astype(float)
        
        ## passoire df creation
        self.df_pass = gpd.read_file(self.file_path + 'passoires-par-iris-v2.gpkg') 
        self.df_pass=self.df_pass[self.df_pass['department'].isin(['75','77','78','91','92','93','94','95'])]
        self.df_pass.drop(['region','Total','city_group'], axis=1, errors='ignore', inplace=True)
        self.df_pass.rename(columns = {'Nombre': 'nb_passoires', 'Taux': 'pourcentage_passoires'}, inplace=True) 
        
    
    def add_subdivisions(self) -> pd.DataFrame:
        """Adds communes/arrondisements to locations
        
        Returns:
            pd.merge: merged original dataframe with ile-de-france geometric data
        """
        self.df['l_codinsee'] = self.df['l_codinsee'].astype(float)
        return pd.merge(self.df, self.idf_reg, on='l_codinsee')
       
    def count_public_transport_spots(self,
                                     distance_to_station: int = 200, 
                                     batch_count: int = 150) -> pd.DataFrame:
        """
        Count the number of public transport spots within 200 meters of each point in df1.
        
        Args:
            subdivisons: split the df into x subdivisions (for memory purposes)
            
        Returns:
            self.df: returns df with transport nearby
        """
        coords2 = np.radians(self.gares[['latitude', 'longitude']].to_numpy())
        lats2, lons2 = coords2[:, 0], coords2[:, 1]
        
        sub_dfs = np.array_split(self.df, batch_count)

        spots = []
        for i in tqdm(range(batch_count)):
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
    
    def add_crimerate(self) -> pd.DataFrame:
        """Add crimerate per commune to original dataset
        
        Returns: 
            df: merged dataset on arrondisement and coddep
        """
        self.df['arrondissement'] = [str(x)[5:7] if str(x).startswith("['75") else np.nan for x in self.df['l_codinsee']]
        self.df['arrondissement'] = self.df['arrondissement'].astype(float)
        return self.df.merge(self.df_crimerate2018, on=['arrondissement','coddep'])
    
    def convert_to_point(self, row) -> Point:
        """Convert lat and long to shapely Point
        
        Args:
            row: tuple of latitude and longitude
            
        Returns:
            Point: shapely Point
        """
        return Point(row["longitude"], row["latitude"])
    
    def add_passoir(self) -> pd.DataFrame:
        """Add passoir thermique values to original dataset
        
        Returns:
            df: geometrically joined dataset with online data
        """
        self.df["lat_long"] = self.df.apply(self.convert_to_point, axis=1)
        self.df = gpd.GeoDataFrame(self.df, geometry="lat_long")
        self.df.crs = {'init': 'epsg:4326'}
        self.df = self.df.to_crs(self.df_pass.crs)
        return gpd.sjoin(self.df_pass, self.df, how='inner', predicate='intersects')
    
    def add_all(self) -> pd.DataFrame:
        print('Adding subdivisions...')
        self.df = self.add_subdivisions()
        print('Counting nearby train stops...')
        self.df = self.count_public_transport_spots()
        print('Adding communal crime rates...')
        self.df = self.add_crimerate()
        print('Counting passoir thermiques...')
        self.df = self.add_passoir()

        return self.df
    

