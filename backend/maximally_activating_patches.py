import pickle
import streamlit as st

from backend.load_file import load_json


@st.cache(allow_output_mutation=True)
def load_activation(filename):
    activation = load_json(filename)
    return activation

@st.cache(allow_output_mutation=True)
def load_dataset(data_index):
    with open(f'./data/preprocessed_image_net/val_data_{data_index}.pkl', 'rb') as file:
        dataset = pickle.load(file)
    return dataset

def load_layer_infos(filename):
    layer_infos = load_json(filename)
    return layer_infos

def get_receptive_field_coordinates(layer_infos, layer_name, idx_x, idx_y):
    """
    layer_name: as in layer_infos keys (eg: 'encoder.stages[0].layers[0]')
    idx_x: integer coordinate of width axis in feature maps. must < n
    idx_y: integer coordinate of height axis in feature maps. must < n
    """
    layer_name = layer_name.replace('.dwconv', '').replace('.layernorm', '')
    layer_name = layer_name.replace('.pwconv1', '').replace('.pwconv2', '').replace('.drop_path', '')
    n = layer_infos[layer_name]['n']
    j = layer_infos[layer_name]['j']
    r = layer_infos[layer_name]['r']
    start = layer_infos[layer_name]['start']
    assert idx_x < n, f'n={n}'
    assert idx_y < n, f'n={n}'

    # image tensor (N, H, W, C) or (N, C, H, W) => image_patch=image[y1:y2, x1:x2]
    center = (start + idx_x*j, start + idx_y*j)
    x1, x2 = (max(center[0]-r/2, 0), max(center[0]+r/2, 0))
    y1, y2 = (max(center[1]-r/2, 0), max(center[1]+r/2, 0))
    x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

    return x1, x2, y1, y2
