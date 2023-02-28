import PIL
from PIL import Image
import numpy as np
from matplotlib import pylab as P
import cv2

import torch
from torch.utils.data import TensorDataset
from torchvision import transforms
import torch.nn.functional as F

from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from torchvex.base import ExplanationMethod
from torchvex.utils.normalization import clamp_quantile

from backend.utils import load_image, load_model
from backend.smooth_grad import generate_smoothgrad_mask

import streamlit as st

IMAGENET_DEFAULT_MEAN = np.asarray(IMAGENET_DEFAULT_MEAN).reshape([1,3,1,1])
IMAGENET_DEFAULT_STD = np.asarray(IMAGENET_DEFAULT_STD).reshape([1,3,1,1])

def deprocess_image(image_inputs):
    return (image_inputs * IMAGENET_DEFAULT_STD + IMAGENET_DEFAULT_MEAN) * 255


def feed_forward(input_image):
    model, feature_extractor = load_model('ConvNeXt')
    inputs = feature_extractor(input_image, do_resize=False, return_tensors="pt")['pixel_values']
    logits = model(inputs).logits
    prediction_prob = F.softmax(logits, dim=-1).max()                   # prediction probability
    # prediction class id, start from 1 to 1000 so it needs to +1 in the end
    prediction_class = logits.argmax(-1).item()                 
    prediction_label = model.config.id2label[prediction_class]  # prediction class label
    return prediction_prob, prediction_class, prediction_label

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient and normalize it
    sign_data_grad = torch.gt(data_grad, 0).type(torch.FloatTensor) * 2.0 - 1.0
    perturbed_image = image + epsilon*sign_data_grad
    return perturbed_image

# perform attack on the model
def perform_attack(input_image, target, epsilon):
    model, feature_extractor = load_model("ConvNeXt")
    # preprocess input image
    inputs = feature_extractor(input_image, do_resize=False, return_tensors="pt")['pixel_values']
    inputs.requires_grad = True
    
    # predict
    logits = model(inputs).logits
    prediction_prob = F.softmax(logits, dim=-1).max()
    prediction_class = logits.argmax(-1).item()
    prediction_label = model.config.id2label[prediction_class]
    
    # Calculate the loss
    loss = F.nll_loss(logits, torch.tensor([target]))

    # Zero all existing gradients
    model.zero_grad()

    # Calculate gradients of model in backward pass
    loss.backward()
    
    # Collect datagrad
    data_grad = inputs.grad.data
    
    # Call FGSM Attack
    perturbed_data = fgsm_attack(inputs, epsilon, data_grad)
    
    # Re-classify the perturbed image
    new_prediction = model(perturbed_data).logits
    new_pred_prob = F.softmax(new_prediction, dim=-1).max()
    new_pred_class = new_prediction.argmax(-1).item()
    new_pred_label = model.config.id2label[new_pred_class]

    return perturbed_data, new_pred_prob.item(), new_pred_class, new_pred_label


def find_smallest_epsilon(input_image, target):
    epsilons = [i*0.001 for i in range(1000)]

    for epsilon in epsilons:
        perturbed_data, new_prob, new_id, new_label = perform_attack(input_image, target, epsilon)
        if new_id != target:
            return perturbed_data, new_prob, new_id, new_label, epsilon
    return None

@st.cache_data
def generate_images(image_id, epsilon=0):
    model, feature_extractor = load_model("ConvNeXt")
    original_image_dict = load_image(image_id)
    image = original_image_dict['image']
    return generate_smoothgrad_mask(
        image, 'ConvNeXt',
        model, feature_extractor, num_samples=10, return_mask=True)
