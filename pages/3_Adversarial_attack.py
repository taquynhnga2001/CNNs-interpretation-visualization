import streamlit as st
import pandas as pd
import numpy as np
import random
from backend.utils import make_grid, load_dataset, load_model, load_image

from backend.smooth_grad import generate_smoothgrad_mask, ShowImage, fig2img, LoadImage, ShowHeatMap, ShowMaskedImage
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch

from matplotlib.backends.backend_agg import RendererAgg

from backend.adversarial_attack import *

_lock = RendererAgg.lock

st.set_page_config(layout='wide')
BACKGROUND_COLOR = '#bcd0e7'
SECONDARY_COLOR = '#bce7db'


st.title('Adversarial Attack')
st.write('How adversarial attacks affect ConvNeXt interpretation?')

imagenet_df = pd.read_csv('./data/ImageNet_metadata.csv')
image_id = None

if 'image_id' not in st.session_state:
    st.session_state.image_id = 0

# def on_change_random_input():
#     st.session_state.image_id = st.session_state.image_id

# ----------------------------- INPUT ----------------------------------
st.header('Input')
input_col_1, input_col_2, input_col_3 = st.columns(3)
# --------------------------- INPUT column 1 ---------------------------
with input_col_1:
    with st.form('image_form'):
        
        # image_id = st.number_input('Image ID: ', format='%d', step=1)
        st.write('**Choose or generate a random image**')
        chosen_image_id_input = st.empty()
        image_id = chosen_image_id_input.number_input('Image ID:', format='%d', step=1, value=st.session_state.image_id)
        
        choose_image_button = st.form_submit_button('Choose the defined image')
        random_id = st.form_submit_button('Generate a random image')

        if random_id:
            image_id = random.randint(0, 50000)
            st.session_state.image_id = image_id
            chosen_image_id_input.number_input('Image ID:', format='%d', step=1, value=st.session_state.image_id)
            
        if choose_image_button:
            image_id = int(image_id)
            st.session_state.image_id = int(image_id)
        # st.write(image_id, st.session_state.image_id)

# ---------------------------- SET UP OUTPUT ------------------------------
epsilon_container = st.empty()
st.header('Output')
st.subheader('Perform attack')

# perform attack container
header_col_1, header_col_2, header_col_3, header_col_4, header_col_5 = st.columns([1,1,1,1,1])
output_col_1, output_col_2, output_col_3, output_col_4, output_col_5 = st.columns([1,1,1,1,1])

# prediction error container
error_container = st.empty()
smoothgrad_header_container = st.empty()

# smoothgrad container
smooth_head_1, smooth_head_2, smooth_head_3, smooth_head_4, smooth_head_5 = st.columns([1,1,1,1,1])
smoothgrad_col_1, smoothgrad_col_2, smoothgrad_col_3, smoothgrad_col_4, smoothgrad_col_5 = st.columns([1,1,1,1,1])

original_image_dict = load_image(st.session_state.image_id)
input_image = original_image_dict['image']
input_label = original_image_dict['label']
input_id = original_image_dict['id']

# ---------------------------- DISPLAY COL 1 ROW 1 ------------------------------
with output_col_1:
    pred_prob, pred_class_id, pred_class_label = feed_forward(input_image)
    # st.write(f'Class ID {input_id} - {input_label}: {pred_prob*100:.3f}% confidence')
    st.image(input_image)
    header_col_1.write(f'Class ID {input_id} - {input_label}: {pred_prob*100:.1f}% confidence')



if pred_class_id != (input_id-1):
    with error_container.container():
        st.write(f'Predicted output: Class ID {pred_class_id} - {pred_class_label} {pred_prob*100:.1f}% confidence')
        st.error('ConvNeXt misclassified the chosen image. Please choose or generate another image.',
            icon = "🚫")

# ----------------------------- INPUT column 2 & 3 ----------------------------
with input_col_2:
    with st.form('epsilon_form'):
        st.write('**Set epsilon or find the smallest epsilon automatically**')
        chosen_epsilon_input = st.empty()
        epsilon = chosen_epsilon_input.number_input('Epsilon:', min_value=0.001, format='%.3f', step=0.001)

        epsilon_button = st.form_submit_button('Choose the defined epsilon')
        find_epsilon = st.form_submit_button('Find the smallest epsilon automatically')


with input_col_3:
    with st.form('iterate_epsilon_form'):
        max_epsilon = st.number_input('Maximum value of epsilon (Optional setting)', value=0.500, format='%.3f')
        step_epsilon = st.number_input('Step (Optional setting)', value=0.001, format='%.3f')
        setting_button = st.form_submit_button('Set iterating mode')


# ---------------------------- DISPLAY COL 2 ROW 1 ------------------------------
if pred_class_id == (input_id-1) and (epsilon_button or find_epsilon or setting_button):
    with output_col_3:
        if epsilon_button:
            perturbed_data, new_prob, new_id, new_label = perform_attack(input_image, input_id-1, epsilon)
        else:
            epsilons = [i*step_epsilon for i in range(1, 1001) if i*step_epsilon <= max_epsilon]
            epsilon_container.progress(0, text='Checking epsilon')
        
            for i, e in enumerate(epsilons):
                print(e)
                
                perturbed_data, new_prob, new_id, new_label = perform_attack(input_image, input_id-1, e)
                epsilon_container.progress(i/len(epsilons), text=f'Checking epsilon={e:.3f}. Confidence={new_prob*100:.1f}%')
                epsilon = e

                if new_id != input_id - 1:
                    epsilon_container.empty()
                    st.balloons()
                    break
                if i == len(epsilons)-1:
                    epsilon_container.error(f'FSGM failed to attack on this image at epsilon={e:.3f}. Set higher maximum value of epsilon or choose another image',
                                            icon = "🚫")

        perturbed_image = deprocess_image(perturbed_data.detach().numpy())[0].astype(np.uint8).transpose(1,2,0)
        perturbed_amount = perturbed_image - input_image
        header_col_3.write(f'Pertubed amount - epsilon={epsilon:.3f}')
        st.image(ShowImage(perturbed_amount))
    
    with output_col_2:
        # st.write('plus sign')
        st.image(LoadImage('frontend/images/plus-sign.png'))
    
    with output_col_4:
        # st.write('equal sign')
        st.image(LoadImage('frontend/images/equal-sign.png'))

    # ---------------------------- DISPLAY COL 5 ROW 1 ------------------------------
    with output_col_5:
        # st.write(f'ID {new_id+1} - {new_label}: {new_prob*100:.3f}% confidence')
        st.image(ShowImage(perturbed_image))
    header_col_5.write(f'Class ID {new_id+1} - {new_label}: {new_prob*100:.1f}% confidence')

    # -------------------------- DISPLAY SMOOTHGRAD ---------------------------
    smoothgrad_header_container.subheader('SmoothGrad visualization')

    with smoothgrad_col_1:
        smooth_head_1.write(f'SmoothGrad before attacked')
        heatmap_image, masked_image, mask = generate_images(st.session_state.image_id, epsilon=0)
        st.image(heatmap_image)
        st.image(masked_image) 
    with smoothgrad_col_3:
        smooth_head_3.write('SmoothGrad after attacked')
        heatmap_image_attacked, masked_image_attacked, attacked_mask= generate_images(st.session_state.image_id, epsilon=epsilon)
        st.image(heatmap_image_attacked)
        st.image(masked_image_attacked)
    
    with smoothgrad_col_2:
        st.image(LoadImage('frontend/images/minus-sign-5.png'))

    with smoothgrad_col_5:
        smooth_head_5.write('SmoothGrad difference')
        difference_mask = abs(attacked_mask-mask)
        st.image(ShowHeatMap(difference_mask))
        masked_image = ShowMaskedImage(difference_mask, perturbed_image)
        st.image(masked_image)

    with smoothgrad_col_4:
        st.image(LoadImage('frontend/images/equal-sign.png'))


