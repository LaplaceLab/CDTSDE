



import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F
from torchvision.transforms import ToTensor, ToPILImage


def adain_color_fix(target: Image, source: Image):
                               
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)

                                           
    result_tensor = adaptive_instance_normalization(target_tensor, source_tensor)

                                  
    to_image = ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))

    return result_image

def wavelet_color_fix(target: Image, source: Image):
                               
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)

                                  
    result_tensor = wavelet_reconstruction(target_tensor, source_tensor)

                                  
    to_image = ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))

    return result_image

def calc_mean_std(feat: Tensor, eps=1e-5):





    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.reshape(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().reshape(b, c, 1, 1)
    feat_mean = feat.reshape(b, c, -1).mean(dim=2).reshape(b, c, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat:Tensor, style_feat:Tensor):






    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def wavelet_blur(image: Tensor, radius: int):


    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
                                                                 
    kernel = kernel[None, None]
                                                 
    kernel = kernel.repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode='replicate')
                       
    output = F.conv2d(image, kernel, groups=3, dilation=radius)
    return output

def wavelet_decomposition(image: Tensor, levels=5):



    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        high_freq += (image - low_freq)
        image = low_freq

    return high_freq, low_freq

def wavelet_reconstruction(content_feat:Tensor, style_feat:Tensor):


    content_high_freq, content_low_freq = wavelet_decomposition(content_feat)
    del content_low_freq
                                                              
    style_high_freq, style_low_freq = wavelet_decomposition(style_feat)
    del style_high_freq
                                                                     
    return content_high_freq + style_low_freq