import streamlit as st
import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import graphviz

from backend.maximally_activating_patches import load_layer_infos, load_activation, get_receptive_field_coordinates
from frontend import on_click_graph
from backend.utils import load_dataset_dict

HIGHTLIGHT_COLOR = '#e7bcc5'
st.set_page_config(layout='wide')

# -------------------------- LOAD DATASET ---------------------------------
dataset_dict = load_dataset_dict()

# -------------------------- LOAD GRAPH -----------------------------------

def load_dot_to_graph(filename):
    dot = graphviz.Source.from_file(filename)
    source_lines = str(dot).splitlines()
    source_lines.pop(0)
    source_lines.pop(-1)
    graph = graphviz.Digraph()
    graph.body += source_lines
    return graph, dot
    
st.title('Maximally activating image patches')
st.write('Visualize image patches that maximize the activation of layers in three models: ConvNeXt, ResNet, MobileNet')

# st.header('ConvNeXt')
convnext_dot_file = './data/dot_architectures/convnext_architecture.dot'
convnext_graph = load_dot_to_graph(convnext_dot_file)[0]

convnext_graph.graph_attr['size'] = '4,40'

# -------------------------- DISPLAY GRAPH -----------------------------------

def chosen_node_text(clicked_node_title):
    clicked_node_title = clicked_node_title.replace('stage ', 'stage_').replace('block ', 'block_')
    stage_id = clicked_node_title.split()[0].split('_')[1] if 'stage' in clicked_node_title else None
    block_id = clicked_node_title.split()[1].split('_')[1] if 'block' in clicked_node_title else None
    layer_id = clicked_node_title.split()[-1]
    
    if 'embeddings' in layer_id:
        display_text = 'Patchify layer'
        activation_key = 'embeddings.patch_embeddings'
    elif 'downsampling' in layer_id:
        display_text = f'Stage {stage_id} > Downsampling layer'
        activation_key = f'encoder.stages[{stage_id}].downsampling_layer[1]'
    else:
        display_text = f'Stage {stage_id} > Block {block_id} > {layer_id} layer'
        activation_key = f'encoder.stages[{int(stage_id)-1}].layers[{int(block_id)-1}].{layer_id}'
    return display_text, activation_key


props = {
    'hightlight_color': HIGHTLIGHT_COLOR,
    'initial_state': {
        'group_1_header': 'Choose an option from group 1',
        'group_2_header': 'Choose an option from group 2'
    }
}


col1, col2 = st.columns((2,5))
col1.markdown("#### Architecture")
col1.write('')
col1.write('Click on a layer below to generate top-k maximally activating image patches')
col1.graphviz_chart(convnext_graph)

with col2:
    st.markdown("#### Output")
    nodes = on_click_graph(key='toggle_buttons', **props)

# -------------------------- DISPLAY OUTPUT -----------------------------------

if nodes != None:
    clicked_node_title = nodes["choice"]["node_title"]
    clicked_node_id = nodes["choice"]["node_id"]
    display_text, activation_key = chosen_node_text(clicked_node_title)
    col2.write(f'**Chosen layer:** {display_text}')
    # col2.write(f'**Activation key:** {activation_key}')

    hightlight_syle = f'''
        <style>
            div[data-stale]:has(iframe) {{
                height: 0;
            }}
            #{clicked_node_id}>polygon {{
                fill: {HIGHTLIGHT_COLOR};
                stroke: {HIGHTLIGHT_COLOR};
            }}
        </style>
    '''
    col2.markdown(hightlight_syle, unsafe_allow_html=True)

    with col2:
        layer_infos = None
        with st.form('top_k_form'):
            activation_path = './data/activation/convnext_activation.json'
            activation = load_activation(activation_path)
            num_channels = activation[activation_key].shape[1]

            top_k = st.slider('Choose K for top-K maximally activating patches', 1,20, value=10)
            channel_start, channel_end = st.slider(
                'Choose channel range of this layer (recommend to choose small range less than 30)',
                1, num_channels, value=(1, 30))
            summit_button = st.form_submit_button('Generate image patches')
            if summit_button:
                
                activation = activation[activation_key][:top_k,:,:]
                layer_infos = load_layer_infos('./data/layer_infos/convnext_layer_infos.json')
                # st.write(channel_start, channel_end)
                # st.write(activation.shape, activation.shape[1])

        if layer_infos != None:
            num_cols, num_rows = top_k, channel_end - channel_start + 1
            # num_rows = activation.shape[1]
            top_k_coor_max_ = activation
            st.markdown(f"#### Top-{top_k} maximally activating image patches of {num_rows} channels ({channel_start}-{channel_end})")

            for row in range(channel_start, channel_end+1):
                if row == channel_start:
                    top_margin = 50
                    fig = make_subplots(
                        rows=1, cols=num_cols, 
                        subplot_titles=tuple([f"#{i+1}" for i in range(top_k)]), shared_yaxes=True)
                else:
                    top_margin = 0
                    fig = make_subplots(rows=1, cols=num_cols)
                for col in range(1, num_cols+1):
                    k, c = col-1, row-1
                    img_index = int(top_k_coor_max_[k, c, 3])
                    activation_value = top_k_coor_max_[k, c, 0]
                    img = dataset_dict[img_index//10_000][img_index%10_000]['image']
                    class_label = dataset_dict[img_index//10_000][img_index%10_000]['label']
                    class_id = dataset_dict[img_index//10_000][img_index%10_000]['id']

                    idx_x, idx_y = top_k_coor_max_[k, c, 1], top_k_coor_max_[k, c, 2]
                    x1, x2, y1, y2 = get_receptive_field_coordinates(layer_infos, activation_key, idx_x, idx_y)
                    img = np.array(img)[y1:y2, x1:x2, :]
                    
                    hovertemplate = f"""Top-{col}<br>Activation value: {activation_value:.5f}<br>Class Label: {class_label}<br>Class id: {class_id}<br>Image id: {img_index}"""
                    fig.add_trace(go.Image(z=img, hovertemplate=hovertemplate), row=1, col=col)
                    fig.update_xaxes(showticklabels=False, showgrid=False)
                    fig.update_yaxes(showticklabels=False, showgrid=False)
                    fig.update_layout(margin={'b':0, 't':top_margin, 'r':0, 'l':0})
                    fig.update_layout(showlegend=False, yaxis_title=row)
                    fig.update_layout(height=100, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    fig.update_layout(hoverlabel=dict(bgcolor="#e9f2f7"))
                st.plotly_chart(fig, use_container_width=True)


else:
    col2.markdown(f'Chosen layer: <code>None</code>', unsafe_allow_html=True)
    col2.markdown("""<style>div[data-stale]:has(iframe) {height: 0};""", unsafe_allow_html=True)
