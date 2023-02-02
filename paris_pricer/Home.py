from __future__ import annotations

import pandas as pd
import streamlit as st

from helpers import Data, Model, Scraper

st.set_page_config(page_title='Real estate price estimation')

st.write('Find out what a place would be worth?')

# Setting up the session_state variables
if 'scraper' not in st.session_state:
    st.session_state['scraper'] = Scraper()

if 'place' not in st.session_state:
    st.session_state['place'] = '78 avenue Raymond Poincaré'

if 'model' not in st.session_state:
    st.session_state['model'] = Model.load_pricing_model()

scraper = st.session_state['scraper']

# Searching for the desired place
scraper.search_place_with_url(st.session_state['place'])

st.text_input(label='', key='place')

# Retrieving the coordinates of the desired place
latitude, longitude = scraper.get_coordinates()

# df = Data.load_data_for_model()
if 'df' not in st.session_state:
    df = Data.load_sample_df()
    st.session_state['df'] = df

# Calculating the distance
df_distance = st.session_state['df'].pipe(Data.calculate_distance, latitude=latitude, longitude=longitude)
model_data = df_distance.loc[[df_distance['distance'].argmin()], :].head(1).reset_index(drop=True)

# Creating the price predictions for now and for in five years
columns = st.columns(3)
with columns[0]:
    current_price = st.session_state['model'].predict(model_data)[0]
    st.metric(label='Current price',
              value=f"{current_price:,.{2}f} €")
with columns[1]:
    model_data_in_five_years = model_data.copy()
    model_data_in_five_years.loc[0, 'anneemut'] += 5
    price_in_five_years = st.session_state['model'].predict(model_data_in_five_years)[0]
    st.metric(label='Price in five years',
              value=f"{price_in_five_years:,.{2}f} €",
              delta=f'{(((price_in_five_years / current_price) - 1) * 100):,.{2}f} %')
with columns[2]:
    ecb_five_year_equivalent = current_price * (1.02312) ** 5
    st.metric(label='ECB bond equivalent',
              value=f"{ecb_five_year_equivalent:,.{2}f} €",
              delta=f'{(((ecb_five_year_equivalent / current_price) - 1) * 100):,.{2}f} %')

# Drawing the map
st.map(pd.DataFrame({'lat': [latitude], 'lon': [longitude]}))
