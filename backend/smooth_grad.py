import PIL
from PIL import Image
import numpy as np
from matplotlib import pylab as P
import cv2

import torch
from torch.utils.data import TensorDataset
from torchvision import transforms

# dirpath_to_modules = './Visual-Explanation-Methods-PyTorch'
# sys.path.append(dirpath_to_modules)

from torchvex.base import ExplanationMethod
from torchvex.utils.normalization import clamp_quantile

def ShowImage(im, title='', ax=None):
    image = np.array(im)
    return image

def ShowGrayscaleImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
    P.title(title)
    return P

def ShowHeatMap(im, title='', ax=None):
    im = im - im.min()
    im = im / im.max()
    im = im.clip(0,1)
    im = np.uint8(im * 255)
    
    im = cv2.resize(im, (224,224))
    image = cv2.resize(im, (224, 224))

    # Apply JET colormap
    color_heatmap = cv2.applyColorMap(image, cv2.COLORMAP_HOT)
    # P.imshow(im, cmap='inferno')
    # P.title(title)
    return color_heatmap
    
def ShowMaskedImage(saliency_map, image, title='', ax=None):
    """ 
    Save saliency map on image.
    
    Args:
        image: Tensor of size (H,W,3)
        saliency_map: Tensor of size (H,W,1)
    """
    
    # if ax is None:
    #     P.figure()
    # P.axis('off')

    saliency_map = saliency_map - saliency_map.min()
    saliency_map = saliency_map / saliency_map.max()
    saliency_map = saliency_map.clip(0,1)
    saliency_map = np.uint8(saliency_map * 255)
    
    saliency_map = cv2.resize(saliency_map, (224,224))
    image = cv2.resize(image, (224, 224))

    # Apply JET colormap
    color_heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_HOT)
    
    # Blend image with heatmap
    img_with_heatmap = cv2.addWeighted(image, 0.4, color_heatmap, 0.6, 0)

    # P.imshow(img_with_heatmap)
    # P.title(title)
    return img_with_heatmap

def LoadImage(file_path):
    im = PIL.Image.open(file_path)
    im = im.resize((224, 224))
    im = np.asarray(im)
    return im


def visualize_image_grayscale(image_3d, percentile=99):
    r"""Returns a 3D tensor as a grayscale 2D tensor.
    This method sums a 3D tensor across the absolute value of axis=2, and then
    clips values at a given percentile.
    """
    image_2d = np.sum(np.abs(image_3d), axis=2)

    vmax = np.percentile(image_2d, percentile)
    vmin = np.min(image_2d)

    return np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)

def visualize_image_diverging(image_3d, percentile=99):
    r"""Returns a 3D tensor as a 2D tensor with positive and negative values.
    """
    image_2d = np.sum(image_3d, axis=2)

    span = abs(np.percentile(image_2d, percentile))
    vmin = -span
    vmax = span

    return np.clip((image_2d - vmin) / (vmax - vmin), -1, 1)


class SimpleGradient(ExplanationMethod):
    def __init__(self, model, create_graph=False,
                 preprocess=None, postprocess=None):
        super().__init__(model, preprocess, postprocess)
        self.create_graph = create_graph

    def predict(self, x):
        return self.model(x)

    @torch.enable_grad()
    def process(self, inputs, target):
        self.model.zero_grad()
        inputs.requires_grad_(True)

        out = self.model(inputs)
        out = out if type(out) == torch.Tensor else out.logits

        num_classes = out.size(-1)
        onehot = torch.zeros(inputs.size(0), num_classes, *target.shape[1:])
        onehot = onehot.to(dtype=inputs.dtype, device=inputs.device)
        onehot.scatter_(1, target.unsqueeze(1), 1)

        grad, = torch.autograd.grad(
            (out*onehot).sum(), inputs, create_graph=self.create_graph
        )

        return grad


class SmoothGradient(ExplanationMethod):
    def __init__(self, model, stdev_spread=0.15, num_samples=25,
                 magnitude=True, batch_size=-1,
                 create_graph=False, preprocess=None, postprocess=None):
        super().__init__(model, preprocess, postprocess)
        self.stdev_spread = stdev_spread
        self.nsample = num_samples
        self.create_graph = create_graph
        self.magnitude = magnitude
        self.batch_size = batch_size
        if self.batch_size == -1:
            self.batch_size = self.nsample

        self._simgrad = SimpleGradient(model, create_graph)

    def process(self, inputs, target):
        self.model.zero_grad()

        maxima = inputs.flatten(1).max(-1)[0]
        minima = inputs.flatten(1).min(-1)[0]

        stdev = self.stdev_spread * (maxima - minima).cpu()
        stdev = stdev.view(inputs.size(0), 1, 1, 1).expand_as(inputs)
        stdev = stdev.unsqueeze(0).expand(self.nsample, *[-1]*4)
        noise = torch.normal(0, stdev)

        target_expanded = target.unsqueeze(0).cpu()
        target_expanded = target_expanded.expand(noise.size(0), -1)

        noiseloader = torch.utils.data.DataLoader(
            TensorDataset(noise, target_expanded), batch_size=self.batch_size
        )

        total_gradients = torch.zeros_like(inputs)
        for noise, t_exp in noiseloader:
            inputs_w_noise = inputs.unsqueeze(0) + noise.to(inputs.device)
            inputs_w_noise = inputs_w_noise.view(-1, *inputs.shape[1:])
            gradients = self._simgrad(inputs_w_noise, t_exp.view(-1))
            gradients = gradients.view(self.batch_size, *inputs.shape)
            if self.magnitude:
                gradients = gradients.pow(2)
            total_gradients = total_gradients + gradients.sum(0)

        smoothed_gradient = total_gradients / self.nsample
        return smoothed_gradient


def feed_forward(model_name, image, model=None, feature_extractor=None):
    if model_name in ['ConvNeXt', 'ResNet']:
        inputs = feature_extractor(image, return_tensors="pt")
        logits = model(**inputs).logits
        prediction_class = logits.argmax(-1).item()
    else:
        transform_images = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        input_tensor = transform_images(image)
        inputs = input_tensor.unsqueeze(0)

        output = model(inputs)
        prediction_class = output.argmax(-1).item()
    #prediction_label = model.config.id2label[prediction_class]
    return inputs, prediction_class

def clip_gradient(gradient):
    gradient = gradient.abs().sum(1, keepdim=True)
    return clamp_quantile(gradient, q=0.99)

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def generate_smoothgrad_mask(image, model_name, model=None, feature_extractor=None, num_samples=25, return_mask=False):
    inputs, prediction_class = feed_forward(model_name, image, model, feature_extractor)

    smoothgrad_gen = SmoothGradient(
        model, num_samples=num_samples, stdev_spread=0.1,
        magnitude=False, postprocess=clip_gradient)

    if type(inputs) != torch.Tensor:
        inputs = inputs['pixel_values']

    smoothgrad_mask = smoothgrad_gen(inputs, prediction_class)
    smoothgrad_mask = smoothgrad_mask[0].numpy()
    smoothgrad_mask = np.transpose(smoothgrad_mask, (1, 2, 0))

    image = np.asarray(image)
    # ori_image = ShowImage(image)
    heat_map_image = ShowHeatMap(smoothgrad_mask)
    masked_image = ShowMaskedImage(smoothgrad_mask, image)

    if return_mask:
        return heat_map_image, masked_image, smoothgrad_mask
    else:
        return heat_map_image, masked_image
