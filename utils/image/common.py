import random
import math

from PIL import Image
import numpy as np
import cv2
import torch
from torch.nn import functional as F


                                                                                         
def center_crop_arr(pil_image, image_size):
                                                                  
                                                                   
                                                           
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


                                                                                         
def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

                                                                  
                                                                   
                                                           
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


                                                                               
def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):




















    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:              
            cv2.flip(img, 1, img)
        if vflip:            
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:              
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:            
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


                                                                                      
def filter2D(img, kernel):





    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    else:
        raise ValueError('Wrong kernel size')

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
                                                   
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)


                                                                                                                       
def rgb2ycbcr_pt(img, y_only=False):











    if y_only:
        weight = torch.tensor([[65.481], [128.553], [24.966]]).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]).to(img)
        bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    out_img = out_img / 255.
    return out_img


def to_pil_image(inputs, mem_order, val_range, channel_order):
                                   
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.cpu().numpy()
    assert isinstance(inputs, np.ndarray)
    
                                                  
    if mem_order in ["hwc", "chw"]:
        inputs = inputs[None, ...]
        mem_order = f"n{mem_order}"
             
    if mem_order == "nchw":
        inputs = inputs.transpose(0, 2, 3, 1)
            
    if channel_order == "bgr":
        inputs = inputs[..., ::-1].copy()
    else:
        assert channel_order == "rgb"
    
    if val_range == "0,1":
        inputs = inputs * 255
    elif val_range == "-1,1":
        inputs = (inputs + 1) * 127.5
    else:
        assert val_range == "0,255"
    
    inputs = inputs.clip(0, 255).astype(np.uint8)
    return [inputs[i] for i in range(len(inputs))]


def put_text(pil_img_arr, text):
    cv_img = pil_img_arr[..., ::-1].copy()
    cv2.putText(cv_img, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return cv_img[..., ::-1].copy()


def auto_resize(img: Image.Image, size: int) -> Image.Image:
    short_edge = min(img.size)
    if short_edge < size:
        r = size / short_edge
        img = img.resize(
            tuple(math.ceil(x * r) for x in img.size), Image.BICUBIC
        )
    else:
                                                   
        img = img.copy()
    return img


def pad(img: np.ndarray, scale: int) -> np.ndarray:
    h, w = img.shape[:2]
    ph = 0 if h % scale == 0 else math.ceil(h / scale) * scale - h
    pw = 0 if w % scale == 0 else math.ceil(w / scale) * scale - w
    return np.pad(
        img, pad_width=((0, ph), (0, pw), (0, 0)), mode="constant",
        constant_values=0
    )
