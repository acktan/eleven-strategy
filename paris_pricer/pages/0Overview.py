from __future__ import annotations

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import os

from helpers import Data, StreamlitHelpers

# Loading the data
if 'df_idf' not in st.session_state:
    st.session_state['df_idf'] = Data.load_shape_file()

if 'df_mut' not in st.session_state:
    # st.session_state['df_mut'] = Data.load_data_for_model()
    st.session_state['df_mut'] = Data.load_sample_df()
    st.session_state['df_mut'] = Data.turn_mutations_df_into_geodf(st.session_state['df_mut'],
                                                                   crs=st.session_state['df_idf'].crs)

df_idf = st.session_state['df_idf']
df_mut = st.session_state['df_mut']

if 'df_idf_mut' not in st.session_state:
    st.session_state['df_idf_mut'] = gpd.sjoin(df_idf, df_mut, how='inner', predicate='intersects')

df_idf_mut = st.session_state['df_idf_mut']
st.selectbox('Select an aggregation method', options=['median', 'mean', 'max', 'min'], key='agg_func')

df_agg = df_idf_mut.groupby('nomcom').agg({'valeurfonc': st.session_state['agg_func']}).reset_index()

df_com = gpd.GeoDataFrame(pd.merge(df_agg, df_idf[['nomcom', 'geometry']], on='nomcom', how='left'))
df_dep = df_idf[['numdep', 'geometry']].dissolve(by='numdep', aggfunc='sum')

# Create the empty map
m = folium.Map(location=[48.856614, 2.3522219],
               zoom_start=10, tiles='cartodbpositron')

# Add the legend to the map
colors = ['#00ae53', '#86dc76', '#daf8aa', '#ffe6a4', '#ff9a61', '#ee0028']

m.get_root().html.add_child(folium.Element(StreamlitHelpers.get_title_html()))
m.get_root().html.add_child(folium.Element(StreamlitHelpers.get_legend_html(df_com, colors=colors)))

# Add the department outline to the map
folium.GeoJson(
    df_dep,
    style_function=lambda x: {
        'color': 'black',
        'weight': 2.5,
        'fillOpacity': 0
    },
    name='Departement').add_to(m)

# Defining the colormap for the legend and the communes
values = np.linspace(df_com['valeurfonc'].min(), df_com['valeurfonc'].max(), num=7)
rounded_vals = np.around(values / 100_000) * 100_000
colormap_dept = cm.StepColormap(colors=colors,
                                vmin=min(df_com['valeurfonc']),
                                vmax=max(df_com['valeurfonc']),
                                index=rounded_vals)

style_function = lambda x: {'fillColor': colormap_dept(x['properties']['valeurfonc']),
                            'color': '',
                            'weight': 0.0001,
                            'fillOpacity': 0.6}

# Add the commune data to the map
folium.GeoJson(df_com,
               style_function=style_function,
               tooltip=folium.GeoJsonTooltip(fields=['nomcom', 'valeurfonc'],
                                             aliases=['Commune/Arrondisement', 'Valeur fonciere'],
                                             localize=False),
               name='Community').add_to(m)

st_folium(m, width=724)
