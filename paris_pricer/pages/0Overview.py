from __future__ import annotations

import pandas as pd
import streamlit as st

import os

from helpers import Data

st.write(os.listdir('.'))

if 'df' in st.session_state:
    df = st.session_state['df']
else:
    # df = Data.load_data_for_model()
    df = Data.load_sample_df()

st.write(df.head())

columns = st.columns(2)
with columns[0]:
    st.selectbox('Which arrondissement are you looking in?', options=df[''])
with columns[1]:
    st.write('How many square meters are looking for?')
