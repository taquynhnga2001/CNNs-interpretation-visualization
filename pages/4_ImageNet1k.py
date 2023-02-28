import streamlit as st
import pandas as pd

from backend.utils import load_dataset, use_container_width_percentage

st.title('ImageNet-1k')
st.markdown('This page shows the summary of 50,000 images in the validation set of [ImageNet-1k](https://huggingface.co/datasets/imagenet-1k)')

# SCREEN_WIDTH, SCREEN_HEIGHT = 2560, 1664

with st.spinner("Loading dataset..."):
    dataset_dict = {}
    for data_index in range(5):
        dataset_dict[data_index] = load_dataset(data_index)
        
imagenet_df = pd.read_csv('./data/ImageNet_metadata.csv')

class_labels = imagenet_df.ClassLabel.unique().tolist()
class_labels.sort()
selected_classes = st.multiselect('Class filter: ', options=['All'] + class_labels)
if not ('All' in selected_classes or len(selected_classes) == 0):
    imagenet_df = imagenet_df[imagenet_df['ClassLabel'].isin(selected_classes)]
# st.write(class_labels)

col1, col2 = st.columns([2, 1])
with col1:
    st.dataframe(imagenet_df)
    use_container_width_percentage(100)

with col2:
    st.text_area('Type anything here to copy later :)')
    image = None
    with st.form("display image"):
        img_index = st.text_input('Image ID to display')
        try:
            img_index = int(img_index)
        except:
            pass

        submitted = st.form_submit_button('Display this image')
        if submitted:
            image = dataset_dict[img_index//10_000][img_index%10_000]['image']
            class_label = dataset_dict[img_index//10_000][img_index%10_000]['label']
            class_id = dataset_dict[img_index//10_000][img_index%10_000]['id']
    if image != None:
        st.image(image)
        st.write('**Class label:** ', class_label)
        st.write('\n**Class id:** ', str(class_id))
