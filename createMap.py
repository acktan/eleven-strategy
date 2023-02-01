import folium
import numpy as np
import pandas as pd
import geopandas as gpd
import branca.colormap as cm
from shapely.geometry import Point
from paris_pricer.dataAugment import DataAugmentation


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

class CreateMap(DataAugmentation):
    """Create HTML map of Ile-de-France housing prices per commune/arrondisement
    Args:
        df: housing input df
    """
    def __init__(self, df):
        super().__init__(DataAugmentation.idf_reg)
        self.df = df
        
    def convert_to_point(self, row) -> Point:
        """Convert lat and long to shapely Point
        
        Args:
            row: tuple of latitude and longitude
            
        Returns:
            Point: shapely Point
        """
        return Point(row["longitude"], row["latitude"])
    
    def combine_df(self,
                   compute_val: str = 'mean') -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Combine housing df and convert to gpd
        
        Args:
            compute_val: what aggregate function to use for communes/arrondisements
        Returns:
            result: groupby aggregate for each commune/arrondisements
            dept_geo: geographic outline of departments
        """
        self.df["lat_long"] = self.df.apply(self.convert_to_point, axis=1)
        print('Converting to gpd...')
        df_points = gpd.GeoDataFrame(self.df, geometry="lat_long")
        df_points.crs = {'init': 'epsg:4326'}
        idf_reg = gpd.GeoDataFrame(DataAugmentation.idf_reg, geometry='geometry')
        df_points = df_points.to_crs(idf_reg.crs)
        
        print('Creating new files...')
        result = gpd.sjoin(idf_reg, df_points, how='inner', predicate='intersects')
        
        if compute_val not in ['mean', 'median', 'min', 'max']:
            raise ValueError('Value computation does not exist')
        grouped = result.groupby('nomcom').agg(compute_val).reset_index()
        
            
        result = pd.merge(grouped[['nomcom', 'valeurfonc']], idf_reg[['nomcom', 'geometry']], how='left', on='nomcom')
        dept_geo = gpd.GeoDataFrame(idf_reg.set_index('numdep')['geometry'], geometry='geometry').dissolve(by='numdep', aggfunc='sum')
        return result, dept_geo
    
    def map(self, 
            result: gpd.GeoDataFrame,
            dept_geo: gpd.GeoDataFrame) -> folium.Map:
        """Create map of Ile-de-France with housing prices
        """
        
        m = folium.Map(
            location=[48.856614, 2.3522219], # coordinates of Paris
            zoom_start = 10, 
            tiles='cartodbpositron')

        colormap_dept = cm.StepColormap(
            colors=['#00ae53', '#86dc76', '#daf8aa',
                    '#ffe6a4', '#ff9a61', '#ee0028'],
            vmin=min(result['valeurfonc']),
            vmax=max(result['valeurfonc']),
            index=np.linspace(result['valeurfonc'].min(), result['valeurfonc'].max(), num=7))
        
        style_function = lambda x: {
            'fillColor': colormap_dept(x['properties']['valeurfonc']),
            'color': '',
            'weight': 0.0001,
            'fillOpacity': 0.6
            }
        
        folium.GeoJson(
            dept_geo,
            style_function = lambda x: {
                'color': 'black',
                'weight': 2.5,
                'fillOpacity': 0
            },
            name='Departement').add_to(m)

        folium.GeoJson(
            gpd.GeoDataFrame(result),
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=['nomcom', 'valeurfonc'],
                aliases=['Commune/Arrondisement', 'Valeur fonciere'],
                localize=False
            ),
            name='Community').add_to(m)
        
        return m
        
        
                
    
    
    
        

