from __future__ import annotations

import pandas as pd
import streamlit as st

from helpers import Data

if 'df' in st.session_state:
    df = st.session_state['df']
else:
    # df = Data.load_data_for_model()
    df = Data.load_sample_df()

st.write('Welcome to the model interpretability page')
