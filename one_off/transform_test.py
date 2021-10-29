import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import albumentations as A
import random
import albumentations.augmentations.transforms as trans

#taken from: https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation
# Function to distort image
def elastic_transform(image, mask, seed):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    alpha, sigma, alpha_affine = image.shape[1] * 6, image.shape[1] * 0.2, image.shape[1] * 0.2
    image = np.concatenate((image, mask[...,None]), axis=2)
    random_state = np.random.RandomState(seed)
    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    image, mask = image[..., 0:3], image[..., 3].astype(np.int8)
    
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    im_merge_t = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    # im_t = im_merge_t[...,0:3]
    # im_mask_t = im_merge_t[...,3].astype(np.int8)
    return im_merge_t, mask


class transform_custom():
    def __init__(self, seed, p=0.1):
        assert 0<=p<=1

        self.elastic = elastic_transform
        self.transform = A.Compose([
            trans.Blur(p=0.5),
            A.OneOf([
                trans.ColorJitter(p = 1),
                A.ShiftScaleRotate(rotate_limit=5, p=1, border_mode=cv2.BORDER_CONSTANT),
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),  
            ], p=0.5),
            trans.GridDropout(p=0.5, random_offset=True, unit_size_min = 2, unit_size_max=20, mask_fill_value = 0),  
            A.pytorch.ToTensorV2()
        ])
        self.p = p
        self.seed = seed
    
    def transform_img(self, image, mask):
        # rand = random.random()
        # if rand <= self.p:
        #     # option for custom transform 
        #     image, mask = self.elastic(image=image, mask=mask, seed=self.seed)
        

        return self.transform(image=image, mask=mask)


