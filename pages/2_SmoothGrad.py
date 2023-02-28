import streamlit as st
import pandas as pd
import numpy as np
import random
from backend.utils import make_grid, load_dataset, load_model, load_images

from backend.smooth_grad import generate_smoothgrad_mask, ShowImage, fig2img
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch

from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock

st.set_page_config(layout='wide')
BACKGROUND_COLOR = '#bcd0e7'


st.title('Feature attribution with SmoothGrad')
st.write('Which features are responsible for the current prediction? ')

imagenet_df = pd.read_csv('./data/ImageNet_metadata.csv')

# --------------------------- LOAD function -----------------------------

# @st.cache(allow_output_mutation=True)
# @st.cache_data
# def load_images(image_ids):
#     images = []
#     for image_id in image_ids:
#         dataset = load_dataset(image_id//10000)
#         images.append(dataset[image_id%10000])
#     return images

# @st.cache(allow_output_mutation=True, suppress_st_warning=True, show_spinner=False)
# @st.cache_resource
# def load_model(model_name):
#     with st.spinner(f"Loading {model_name} model! This process might take 1-2 minutes..."):
#         if model_name == 'ResNet':
#             model_file_path = 'microsoft/resnet-50'
#             feature_extractor = AutoFeatureExtractor.from_pretrained(model_file_path, crop_pct=1.0)
#             model = AutoModelForImageClassification.from_pretrained(model_file_path)
#             model.eval()
#         elif model_name == 'ConvNeXt':
#             model_file_path = 'facebook/convnext-tiny-224'
#             feature_extractor = AutoFeatureExtractor.from_pretrained(model_file_path, crop_pct=1.0)
#             model = AutoModelForImageClassification.from_pretrained(model_file_path)
#             model.eval()
#         else:
#             model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
#             model.eval()
#             feature_extractor = None
#     return model, feature_extractor

images = []
image_ids = []
# INPUT ------------------------------
st.header('Input')
with st.form('smooth_grad_form'):
    st.markdown('**Model and Input Setting**')
    selected_models = st.multiselect('Model', options=['ConvNeXt', 'ResNet', 'MobileNet'])
    selected_image_set = st.selectbox('Image set', ['User-defined set', 'Random set'])
    
    summit_button = st.form_submit_button('Set')
    if summit_button:
        setting_container = st.container()
        # for id in image_ids:
        #     images = load_images(image_ids)

with st.form('2nd_form'):
    st.markdown('**Image set setting**')
    if selected_image_set == 'Random set':
            no_images = st.slider('Number of images', 1, 50, value=10)
            image_ids = random.sample(list(range(50_000)), k=no_images)
    else:
        text = st.text_area('Specific Image IDs', value='0')
        image_ids = list(map(lambda x: int(x.strip()), text.split(',')))

    run_button = st.form_submit_button('Display output')
    if run_button:
        for id in image_ids:
            images = load_images(image_ids)
    
st.header('Output')

models = {}
feature_extractors = {}
        
for i, model_name in enumerate(selected_models):
    models[model_name], feature_extractors[model_name] = load_model(model_name)


# DISPLAY ----------------------------------
header_cols = st.columns([1, 1] + [2]*len(selected_models))
header_cols[0].markdown(f'<div style="text-align: center;margin-bottom: 10px;background-color:{BACKGROUND_COLOR};"><b>Image ID</b></div>', unsafe_allow_html=True)
header_cols[1].markdown(f'<div style="text-align: center;margin-bottom: 10px;background-color:{BACKGROUND_COLOR};"><b>Original Image</b></div>', unsafe_allow_html=True)
for i, model_name in enumerate(selected_models):
    header_cols[i + 2].markdown(f'<div style="text-align: center;margin-bottom: 10px;background-color:{BACKGROUND_COLOR};"><b>{model_name}</b></div>', unsafe_allow_html=True)

grids = make_grid(cols=2+len(selected_models)*2, rows=len(image_ids)+1)
# grids[0][0].write('Image ID')
# grids[0][1].write('Original image')

# for i, model_name in enumerate(selected_models):
#     models[model_name], feature_extractors[model_name] = load_model(model_name)


# @st.cache(allow_output_mutation=True)
@st.cache_data
def generate_images(image_id, model_name):
    j = image_ids.index(image_id)
    image = images[j]['image']
    return generate_smoothgrad_mask(
        image, model_name,
        models[model_name], feature_extractors[model_name], num_samples=10)

with _lock:
    for j, (image_id, image_dict) in enumerate(zip(image_ids, images)):
        grids[j][0].write(f'{image_id}. {image_dict["label"]}')
        image = image_dict['image']
        ori_image = ShowImage(np.asarray(image))
        grids[j][1].image(ori_image)

        for i, model_name in enumerate(selected_models):
            # ori_image, heatmap_image, masked_image = generate_smoothgrad_mask(image,
            # model_name, models[model_name], feature_extractors[model_name], num_samples=10)
            heatmap_image, masked_image = generate_images(image_id, model_name)
            # grids[j][1].image(ori_image)
            grids[j][i*2+2].image(heatmap_image)
            grids[j][i*2+3].image(masked_image)