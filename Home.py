import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import streamlit as st
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch

st.set_page_config(layout='wide')
st.title('About')

st.write('Loaded 3 models')


