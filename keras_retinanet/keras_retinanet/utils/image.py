"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import division
import sys
import os

if not __package__:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    import keras_retinanet.utils
    __package__ = "keras_retinanet.utils"

import numpy as np
import cv2
import imghdr
from PIL import Image, ImageOps
from functools import partial

from .transform import change_transform_origin, _sample_discrete


def is_image(file_path):
    '''
    Checks whether the input file is an image according to this list:
    https://docs.python.org/3/library/imghdr.html
    '''
    try:
        return type(imghdr.what(file_path)) is str
    except:
        return False


def read_image_bgr(path):
    """ Read an image in BGR format.

    Args
        path: Path to the image.
    """
    # We deliberately don't use cv2.imread here, since it gives no feedback on errors while reading the image.
    image = np.ascontiguousarray(Image.open(path).convert('RGB'))
    return image[:, :, ::-1]


def preprocess_image_caffe_fast(x):
    """ 
    Same as `preprocess_image(x, mode='caffe')`
    but avoids the expensive, explicit copy casting operation (astype).
    """
    preprocessed_float32 = np.empty(x.shape, dtype=np.float32)
    imagenet_mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    np.subtract(x, imagenet_mean, out=preprocessed_float32, casting="unsafe")
    return preprocessed_float32


def preprocess_image(x, mode='caffe'):
    """ Preprocess an image by subtracting the ImageNet mean.

    Args
        x: np.array of shape (None, None, 3) or (3, None, None).
        mode: One of "caffe" or "tf".
            - caffe: will zero-center each color channel with
                respect to the ImageNet dataset, without scaling.
            - tf: will scale pixels between -1 and 1, sample-wise.

    Returns
        The input with the ImageNet mean subtracted.
    """
    # mostly identical to "https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py"
    # except for converting RGB -> BGR since we assume BGR already

    # covert always to float32 to keep compatibility with opencv
    x = x.astype(np.float32)

    if mode == 'tf':
        x /= 127.5
        x -= 1.
    elif mode == 'caffe':
        x /= 257. # uint16 input
        x -= [103.939, 116.779, 123.68]
    elif mode == 'torch':
        x /= 255.
        x -= [0.485, 0.456, 0.406]
        x /= [0.229, 0.224, 0.225]

    return x


def adjust_transform_for_image(transform, image, relative_translation):
    """ Adjust a transformation for a specific image.

    The translation of the matrix will be scaled with the size of the image.
    The linear part of the transformation will adjusted so that the origin of the transformation will be at the center of the image.
    """
    height, width, channels = image.shape

    result = transform

    # Scale the translation with the image size if specified.
    if relative_translation:
        result[0:2, 2] *= [width, height]

    # Move the origin of transformation.
    result = change_transform_origin(transform, (0.5 * width, 0.5 * height))

    return result


class TransformParameters:
    """ Struct holding parameters determining how to apply a transformation to an image.

    Args
        fill_mode:             One of: 'constant', 'nearest', 'reflect', 'wrap'
        interpolation:         One of: 'nearest', 'linear', 'cubic', 'area', 'lanczos4'
        cval:                  Fill value to use with fill_mode='constant'
        relative_translation:  If true (the default), interpret translation as a factor of the image size.
                               If false, interpret it as absolute pixels.
    """
    def __init__(
        self,
        fill_mode            = 'nearest',
        interpolation        = 'linear',
        cval                 = 0,
        relative_translation = True,
    ):
        self.fill_mode            = fill_mode
        self.cval                 = cval
        self.interpolation        = interpolation
        self.relative_translation = relative_translation

    def cvBorderMode(self):
        if self.fill_mode == 'constant':
            return cv2.BORDER_CONSTANT
        if self.fill_mode == 'nearest':
            return cv2.BORDER_REPLICATE
        if self.fill_mode == 'reflect':
            return cv2.BORDER_REFLECT_101
        if self.fill_mode == 'wrap':
            return cv2.BORDER_WRAP

    def cvInterpolation(self):
        if self.interpolation == 'nearest':
            return cv2.INTER_NEAREST
        if self.interpolation == 'linear':
            return cv2.INTER_LINEAR
        if self.interpolation == 'cubic':
            return cv2.INTER_CUBIC
        if self.interpolation == 'area':
            return cv2.INTER_AREA
        if self.interpolation == 'lanczos4':
            return cv2.INTER_LANCZOS4


def apply_transform(matrix, image, params):
    """
    Apply a transformation to an image.

    The origin of transformation is at the top left corner of the image.

    The matrix is interpreted such that a point (x, y) on the original image is moved to transform * (x, y) in the generated image.
    Mathematically speaking, that means that the matrix is a transformation from the transformed image space to the original image space.

    Args
      matrix: A homogeneous 3 by 3 matrix holding representing the transformation to apply.
      image:  The image to transform.
      params: The transform parameters (see TransformParameters)
    """
    output = cv2.warpAffine(
        image,
        matrix[:2, :],
        dsize       = (image.shape[1], image.shape[0]),
        flags       = params.cvInterpolation(),
        borderMode  = params.cvBorderMode(),
        borderValue = params.cval,
    )
    return output


def compute_resize_scale(image_shape, min_side=800, max_side=1333):
    """ Compute an image scale such that the image size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resizing scale.
    """
    (rows, cols, _) = image_shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    return scale


def resize_image(img, min_side=800, max_side=1333):
    """ Resize an image such that the size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resized image.
    """
    # compute scale to resize the image
    scale = compute_resize_scale(img.shape, min_side=min_side, max_side=max_side)

    # resize the image with the computed scale
    img = cv2.resize(img, None, fx=scale, fy=scale)

    return img, scale


def _uniform(val_range):
    """ Uniformly sample from the given range.

    Args
        val_range: A pair of lower and upper bound.
    """
    return np.random.uniform(val_range[0], val_range[1])


def _check_range(val_range, min_val=None, max_val=None):
    """ Check whether the range is a valid range.

    Args
        val_range: A pair of lower and upper bound.
        min_val: Minimal value for the lower bound.
        max_val: Maximal value for the upper bound.
    """
    if val_range[0] > val_range[1]:
        raise ValueError('interval lower bound > upper bound')
    if min_val is not None and val_range[0] < min_val:
        raise ValueError('invalid interval lower bound')
    if max_val is not None and val_range[1] > max_val:
        raise ValueError('invalid interval upper bound')


def _clip(image):
    """
    Clip and convert an image to np.uint8.

    Args
        image: Image to clip.
    """
    return np.clip(image, 0, 255).astype(np.uint8)


class VisualEffect:
    """ Struct holding parameters and applying image color transformation.

    Args
        contrast_factor:   A factor for adjusting contrast. Should be between 0 and 3.
        brightness_delta:  Brightness offset between -1 and 1 added to the pixel values.
        hue_delta:         Hue offset between -1 and 1 added to the hue channel.
        saturation_factor: A factor multiplying the saturation values of each pixel.
    """

    def __init__(
        self,
        contrast_factor=1,
        brightness_delta=0,
        hue_delta=0,
        saturation_factor=1,
        color_factor=1,
        noise_sigma=0,
        equalize_chance=0
    ):
        self.contrast_factor = contrast_factor
        self.brightness_delta = brightness_delta
        self.hue_delta = hue_delta
        self.saturation_factor = saturation_factor
        self.color_factor = color_factor
        self.noise_sigma = noise_sigma
        self.equalize_chance = equalize_chance

    def __call__(self, image):
        """ Apply a visual effect on the image.

        Args
            image: Image to adjust
        """

        if self.contrast_factor != 1:
            image = adjust_contrast(image, self.contrast_factor)
        if self.brightness_delta != 0:
            image = adjust_brightness(image, self.brightness_delta)

        if self.hue_delta != 0 or self.saturation_factor != 1:

            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            if self.hue_delta:
                image = adjust_hue(image, self.hue_delta)
            if self.saturation_factor:
                image = adjust_saturation(image, self.saturation_factor)

            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        
        if self.color_factor != 1:
            image = adjust_color(image, self.color_factor)

        image = equalize_hist(image, self.equalize_chance)

        if self.noise_sigma != 0:
            image = add_noise(image, self.noise_sigma)

        return image


def random_visual_effect_generator(
    contrast_range=(1., 1.),
    brightness_range=(0., 0.),
    hue_range=(0., 0.),
    saturation_range=(1., 1.),
    color_range=(1., 1.),
    noise_range=(0., 0.),
    equalize_chance=0,
    max_effects=1,
):
    """ Generate visual effect parameters uniformly sampled from the given intervals.

    Args
        contrast_factor:   A factor interval for adjusting contrast. Should be between 0 and 3.
        brightness_delta:  An interval between -1 and 1 for the amount added to the pixels.
        hue_delta:         An interval between -1 and 1 for the amount added to the hue channel.
                           The values are rotated if they exceed 180.
        saturation_factor: An interval for the factor multiplying the saturation values of each
                           pixel.
        color_range:       Color balance factor range
        equalize_chance:   Chance (0 to 1) to perform histogram equalization
        max_effects:       maximum number of effects to apply at once
    """
    _check_range(contrast_range, 0)
    _check_range(brightness_range, -1, 1)
    _check_range(hue_range, -1, 1)
    _check_range(saturation_range, 0)
    _check_range(color_range, 0)
    _check_range(noise_range, 0)

    _FACTORS = (
        "contrast_factor",
        "brightness_delta",
        # "hue_delta",
        # "saturation_factor",
        "color_factor",
        "noise_sigma",
        "equalize_chance"
    )

    _SAMPLERS = (
        partial(_sample_discrete, *contrast_range, neutral_value=1),
        partial(_sample_discrete, *brightness_range),
        # partial(_sample_discrete, *hue_range),
        # partial(_sample_discrete, *saturation_range, neutral_value=1),
        partial(_sample_discrete, *color_range, neutral_value=1),
        partial(_sample_discrete, *noise_range),
        lambda: equalize_chance
    )

    _NUM_EFFECTS = len(_FACTORS)

    assert max_effects in range(1, _NUM_EFFECTS + 1), "Invalid max number of effects given" 

    def _generate():
        while True:
            vis_effect = VisualEffect()
            effect_indices = np.random.randint(0, _NUM_EFFECTS, max_effects+1)[:-1]
            for i in effect_indices:
                setattr(vis_effect, _FACTORS[i], _SAMPLERS[i]())
            yield vis_effect

            # yield VisualEffect(
            #     contrast_factor=_sample_discrete(*contrast_range, neutral_value=1),
            #     brightness_delta=_sample_discrete(*brightness_range),
            #     hue_delta=_sample_discrete(*hue_range),
            #     saturation_factor=_sample_discrete(*saturation_range, neutral_value=1),
            #     color_factor=_sample_discrete(*color_range, neutral_value=1),
            #     equalize_chance=equalize_chance
            # )

    return _generate()


def adjust_contrast(image, factor):
    """ Adjust contrast of an image.

    Args
        image: Image to adjust.
        factor: A factor for adjusting contrast.
    """
    mean = image.mean(axis=0).mean(axis=0)
    return _clip((image - mean) * factor + mean)


def adjust_brightness(image, delta):
    """ Adjust brightness of an image

    Args
        image: Image to adjust.
        delta: Brightness offset between -1 and 1 added to the pixel values.
    """
    return _clip(image + delta * 255)


def adjust_hue(image, delta):
    """ Adjust hue of an image.

    Args
        image: Image to adjust.
        delta: An interval between -1 and 1 for the amount added to the hue channel.
               The values are rotated if they exceed 180.
    """
    image[..., 0] = np.mod(image[..., 0] + delta * 180, 180)
    return image


def adjust_saturation(image, factor):
    """ Adjust saturation of an image.

    Args
        image: Image to adjust.
        factor: An interval for the factor multiplying the saturation values of each pixel.
    """
    image[..., 1] = np.clip(image[..., 1] * factor, 0 , 255)
    return image


def equalize_hist(image, prob):
    """ Equalize image pixel histogram with probability _prob_
    Args
        image: Image to adjust.
        prob: float between [0, 1] denoting probability to take this action
    """
    if _uniform((0, 1)) < prob:
        im = Image.fromarray(image)
        image = np.array(ImageOps.equalize(im))
    return image


def adjust_color(image, factor):
    """ Adjust the color balance of the image
    Args
        image: Image to adjust.
        factor: float between [0, 1] denoting magnitude of color operation,
                0 returns gray image, 1 returns original image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[..., None] * np.ones(3, dtype=np.uint8)
    image = cv2.addWeighted(image, factor, gray, 1 - factor, 0)
    return np.clip(image, 0, 255)


def add_noise(image, sigma):
    """ Add Gaussian noise to the image
    Args
        image: Image to adjust.
        sigma: positive float denoting magnitude of noise (std)
    """
    image = image.astype(np.float64) + sigma * np.random.randn(*image.shape)
    return _clip(image)