# -*- coding: utf-8 -*-
import math
import numbers
import random
import numpy as np
import torchvision.transforms.functional as tf

from PIL import Image, ImageOps


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations
        self.PIL2Numpy = False

    def __call__(self, img, mask1,mask2,mask3,mask4,mask5):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img, mode="RGB")
            mask1 = Image.fromarray(mask1, mode="L")
            mask2 = Image.fromarray(mask2, mode="L")
            mask3 = Image.fromarray(mask3, mode="L")
            mask4 = Image.fromarray(mask4, mode="L")
            mask5 = Image.fromarray(mask5, mode="L")
            self.PIL2Numpy = True
        #print(img.size)
        #print(mask.size)
        #assert img.size == mask.size
        for a in self.augmentations:
            img, mask1,mask2,mask3,mask4,mask5 = a(img, mask1,mask2,mask3,mask4,mask5)

        if self.PIL2Numpy:
            img, mask1,mask2,mask3,mask4,mask5 = np.array(img), np.array(mask1, dtype=np.uint8), np.array(mask2, dtype=np.uint8),\
            np.array(mask3, dtype=np.uint8),np.array(mask4, dtype=np.uint8),np.array(mask5, dtype=np.uint8)

        return img, mask1,mask2,mask3,mask4,mask5


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask1,mask2,mask3,mask4,mask5):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask1 = ImageOps.expand(mask1, border=self.padding, fill=0)
            mask2 = ImageOps.expand(mask2, border=self.padding, fill=0)
            mask3 = ImageOps.expand(mask3, border=self.padding, fill=0)
            mask4 = ImageOps.expand(mask4, border=self.padding, fill=0)
            mask5 = ImageOps.expand(mask5, border=self.padding, fill=0)

        #assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return (img.resize((tw, th), Image.BILINEAR), mask1.resize((tw, th), Image.NEAREST),mask2.resize((tw, th), Image.NEAREST),\
                mask3.resize((tw, th), Image.NEAREST),mask4.resize((tw, th), Image.NEAREST),mask5.resize((tw, th), Image.NEAREST))

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return (img.crop((x1, y1, x1 + tw, y1 + th)), mask1.crop((x1, y1, x1 + tw, y1 + th)),mask2.crop((x1, y1, x1 + tw, y1 + th)),
            mask3.crop((x1, y1, x1 + tw, y1 + th)),mask4.crop((x1, y1, x1 + tw, y1 + th)),mask5.crop((x1, y1, x1 + tw, y1 + th)))


class AdjustGamma(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, img, mask1,mask2,mask3,mask4,mask5):
        #assert img.size == mask.size
        return tf.adjust_gamma(img, random.uniform(1, 1 + self.gamma)), mask1,mask2,mask3,mask4,mask5


class AdjustSaturation(object):
    def __init__(self, saturation):
        self.saturation = saturation

    def __call__(self, img, mask1,mask2,mask3,mask4,mask5):
        #assert img.size == mask.size
        return (
            tf.adjust_saturation(img, random.uniform(1 - self.saturation, 1 + self.saturation)),
            mask1,mask2,mask3,mask4,mask5,
        )


class AdjustHue(object):
    def __init__(self, hue):
        self.hue = hue

    def __call__(self, img, mask1,mask2,mask3,mask4,mask5):
        #assert img.size == mask.size
        return tf.adjust_hue(img, random.uniform(-self.hue, self.hue)), mask1,mask2,mask3,mask4,mask5


class AdjustBrightness(object):
    def __init__(self, bf):
        self.bf = bf

    def __call__(self, img, mask1,mask2,mask3,mask4,mask5):
        #assert img.size == mask.size
        return tf.adjust_brightness(img, random.uniform(1 - self.bf, 1 + self.bf)), mask1,mask2,mask3,mask4,mask5


class AdjustContrast(object):
    def __init__(self, cf):
        self.cf = cf

    def __call__(self, img, mask1,mask2,mask3,mask4,mask5):
        #assert img.size == mask.size
        return tf.adjust_contrast(img, random.uniform(1 - self.cf, 1 + self.cf)), mask1,mask2,mask3,mask4,mask5


# class CenterCrop(object):
#     def __init__(self, size):
#         if isinstance(size, numbers.Number):
#             self.size = (int(size), int(size))
#         else:
#             self.size = size

#     def __call__(self, img, mask):
#         #assert img.size == mask.size
#         w, h = img.size
#         th, tw = self.size
#         x1 = int(round((w - tw) / 2.0))
#         y1 = int(round((h - th) / 2.0))
#         return (img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th)))


class RandomHorizontallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask1,mask2,mask3,mask4,mask5):
        if random.random() < self.p:
            return (img.transpose(Image.FLIP_LEFT_RIGHT), mask1.transpose(Image.FLIP_LEFT_RIGHT),mask2.transpose(Image.FLIP_LEFT_RIGHT),\
                mask3.transpose(Image.FLIP_LEFT_RIGHT),mask4.transpose(Image.FLIP_LEFT_RIGHT),mask5.transpose(Image.FLIP_LEFT_RIGHT))
        return img, mask1,mask2,mask3,mask4,mask5


class RandomVerticallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask1,mask2,mask3,mask4,mask5):
        if random.random() < self.p:
            return (img.transpose(Image.FLIP_TOP_BOTTOM), mask1.transpose(Image.FLIP_TOP_BOTTOM),mask2.transpose(Image.FLIP_TOP_BOTTOM),\
                mask3.transpose(Image.FLIP_TOP_BOTTOM),mask4.transpose(Image.FLIP_TOP_BOTTOM),mask5.transpose(Image.FLIP_TOP_BOTTOM))
        return img, mask1,mask2,mask3,mask4,mask5


# class FreeScale(object):
#     def __init__(self, size):
#         self.size = tuple(reversed(size))  # size: (h, w)

#     def __call__(self, img, mask):
#         assert img.size == mask.size
#         return (img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST))


# class RandomTranslate(object):
#     def __init__(self, offset):
#         # tuple (delta_x, delta_y)
#         self.offset = offset

#     def __call__(self, img, mask):
#         assert img.size == mask.size
#         x_offset = int(2 * (random.random() - 0.5) * self.offset[0])
#         y_offset = int(2 * (random.random() - 0.5) * self.offset[1])

#         x_crop_offset = x_offset
#         y_crop_offset = y_offset
#         if x_offset < 0:
#             x_crop_offset = 0
#         if y_offset < 0:
#             y_crop_offset = 0

#         cropped_img = tf.crop(
#             img,
#             y_crop_offset,
#             x_crop_offset,
#             img.size[1] - abs(y_offset),
#             img.size[0] - abs(x_offset),
#         )

#         if x_offset >= 0 and y_offset >= 0:
#             padding_tuple = (0, 0, x_offset, y_offset)

#         elif x_offset >= 0 and y_offset < 0:
#             padding_tuple = (0, abs(y_offset), x_offset, 0)

#         elif x_offset < 0 and y_offset >= 0:
#             padding_tuple = (abs(x_offset), 0, 0, y_offset)

#         elif x_offset < 0 and y_offset < 0:
#             padding_tuple = (abs(x_offset), abs(y_offset), 0, 0)

#         return (
#             tf.pad(cropped_img, padding_tuple, padding_mode="reflect"),
#             tf.affine(
#                 mask,
#                 translate=(-x_offset, -y_offset),
#                 scale=1.0,
#                 angle=0.0,
#                 shear=0.0,
#                 fillcolor=250,
#             ),
        # )


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask1,mask2,mask3,mask4,mask5):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return (
            tf.affine(
                img,
                translate=(0, 0),
                scale=1.0,
                angle=rotate_degree,
                resample=Image.BILINEAR,
                fillcolor=(0, 0, 0),
                shear=0.0,
            ),
            tf.affine(
                mask1,
                translate=(0, 0),
                scale=1.0,
                angle=rotate_degree,
                resample=Image.NEAREST,
                fillcolor=0,
                shear=0.0,
            ),
            tf.affine(
                mask2,
                translate=(0, 0),
                scale=1.0,
                angle=rotate_degree,
                resample=Image.NEAREST,
                fillcolor=0,
                shear=0.0,
            ),
            tf.affine(
                mask3,
                translate=(0, 0),
                scale=1.0,
                angle=rotate_degree,
                resample=Image.NEAREST,
                fillcolor=0,
                shear=0.0,
            ),
            tf.affine(
                mask4,
                translate=(0, 0),
                scale=1.0,
                angle=rotate_degree,
                resample=Image.NEAREST,
                fillcolor=0,
                shear=0.0,
            ),
            tf.affine(
                mask5,
                translate=(0, 0),
                scale=1.0,
                angle=rotate_degree,
                resample=Image.NEAREST,
                fillcolor=0,
                shear=0.0,
            ),
        )


# class Scale(object):
#     def __init__(self, size):
#         self.size = size

#     def __call__(self, img, mask):
#         #assert img.size == mask.size
#         w, h = img.size
#         if (w >= h and w == self.size) or (h >= w and h == self.size):
#             return img, mask
#         if w > h:
#             ow = self.size
#             oh = int(self.size * h / w)
#             return (img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST))
#         else:
#             oh = self.size
#             ow = int(self.size * w / h)
#             return (img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST))


# class RandomSizedCrop(object):
#     def __init__(self, size):
#         self.size = size

#     def __call__(self, img, mask):
#         assert img.size == mask.size
#         for attempt in range(10):
#             area = img.size[0] * img.size[1]
#             target_area = random.uniform(0.45, 1.0) * area
#             aspect_ratio = random.uniform(0.5, 2)

#             w = int(round(math.sqrt(target_area * aspect_ratio)))
#             h = int(round(math.sqrt(target_area / aspect_ratio)))

#             if random.random() < 0.5:
#                 w, h = h, w

#             if w <= img.size[0] and h <= img.size[1]:
#                 x1 = random.randint(0, img.size[0] - w)
#                 y1 = random.randint(0, img.size[1] - h)

#                 img = img.crop((x1, y1, x1 + w, y1 + h))
#                 mask = mask.crop((x1, y1, x1 + w, y1 + h))
#                 assert img.size == (w, h)

#                 return (
#                     img.resize((self.size, self.size), Image.BILINEAR),
#                     mask.resize((self.size, self.size), Image.NEAREST),
#                 )

#         # Fallback
#         scale = Scale(self.size)
#         crop = CenterCrop(self.size)
#         return crop(*scale(img, mask))