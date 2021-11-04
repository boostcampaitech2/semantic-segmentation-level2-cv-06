import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import albumentations as A
import albumentations.augmentations.transforms as trans
from PIL import Image, ImageOps, ImageEnhance
from albumentations.core.transforms_interface import ImageOnlyTransform

from datasets.copy_paste import CopyPaste

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


class transform_transunet():
    def __init__(self, seed, p=0.5, scale = 2):

        assert 0<=p<=1

        self.scale = scale if scale else 1
        self.elastic = elastic_transform
        self.transform = A.Compose([
            trans.Blur(p=p),
            trans.ToGray(p = p),
            A.ShiftScaleRotate(rotate_limit=15, p=p, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p = p),
            A.Normalize(),
            A.pytorch.ToTensorV2()
        ])
        self.norm_totensor = A.Compose([
            A.Normalize(),
            A.pytorch.ToTensorV2()
        ])
        self.p = p
        self.seed = seed
    
    def transform_img(self, image, mask):
        image = cv2.resize(image, (0,0), fx =self.scale, fy =self.scale, interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
        return self.transform(image=image, mask=mask)


    def val_transform_img(self, image, mask):
        image = cv2.resize(image, (0,0), fx =self.scale, fy =self.scale, interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
        return self.norm_totensor(image=image, mask=mask)

    def test_transform_img(self, image):
        image = cv2.resize(image, (0,0), fx =self.scale, fy =self.scale, interpolation=cv2.INTER_LINEAR).astype(np.float32)
        return self.norm_totensor(image=image)


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
    Returns:
    An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
    Returns:
    A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


augmentations = [
    autocontrast, equalize, posterize, solarize, 
    rotate, shear_x, shear_y,
    translate_x, translate_y
]

augmentations_all = [
    autocontrast, equalize, posterize, solarize, 
    rotate, shear_x, shear_y,
    translate_x, translate_y, 
    color, contrast, brightness, sharpness
]

def normalize(image):
    """Normalize input image channel-wise to zero mean and unit variance."""
    return image - 127


def apply_op(image, op, severity):
    image = image * 255
    image = image.astype(np.uint8)
    
    pil_img = Image.fromarray(image)  # Convert to PIL.Image
    pil_img = op(pil_img, severity)

    return np.asarray(pil_img)


def augment_and_mix(image, severity=3, width=3, depth=-1, alpha=1.):
    """Perform AugMix augmentations and compute mixture.
    Args:
    image: Raw input image as float32 np.ndarray of shape (h, w, c)
    severity: Severity of underlying augmentation operators (between 1 to 10).
    width: Width of augmentation chain
    depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
      from [1, 3]
    alpha: Probability coefficient for Beta and Dirichlet distributions.
    Returns:
    mixed: Augmented and mixed image.
    """
    ws = np.float32(
      np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

    mix = np.zeros_like(image).astype(np.float32)
    for i in range(width):
        image_aug = image.copy()
        depth = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(augmentations_all)
            image_aug = apply_op(image_aug, op, severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * image_aug

    mixed = m * (image * 255) + (1 - m) * mix
    return mixed  / 255


class RandomAugMix(ImageOnlyTransform):
    def __init__(self, severity=3, width=3, depth=-1, alpha=1., always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.severity = severity
        self.width = width
        self.depth = depth
        self.alpha = alpha

    def apply(self, image, **params):
        image = augment_and_mix(
            image,
            self.severity,
            self.width,
            self.depth,
            self.alpha
        )
        return image


class transform_augmix():
    def __init__(self, seed, p=0.5, scale = None):
        self.transform = A.Compose([
            RandomAugMix(severity=3, width=14, alpha=1., p=1),
            A.pytorch.ToTensorV2()
        ])

        self.to_tensor = A.Compose([A.pytorch.ToTensorV2()])
    def transform_img(self, image, mask):
        return self.transform(image=image, mask=mask)


    def val_transform_img(self, image, mask):
        return self.to_tensor(image=image, mask=mask)

    def test_transform_img(self, image):
        return self.to_tensor(image=image)


class transform_copypaste():
    def __init__(self, seed, p=0.5, scale = None):
        self.transform = A.Compose([
        # A.RandomScale(scale_limit=(-0.9, 1), p=1), #LargeScaleJitter from scale of 0.1 to 2
        # A.PadIfNeeded(512, 512, border_mode=0), #pads with image in the center, not the top left like the paper
        # A.RandomCrop(512, 512),
        CopyPaste(blend=True, sigma=1, pct_objects_paste=0.4, p=1.), #pct_objects_paste is a guess
        A.pytorch.ToTensorV2()], bbox_params=A.BboxParams(format="coco", min_visibility=0.05)
        )

        self.to_tensor = A.Compose([A.pytorch.ToTensorV2()])
    def transform_img(self, image, mask):
        return self.transform(image=image, mask=mask)

    def val_transform_img(self, image, mask):
        return self.to_tensor(image=image, mask=mask)

    def test_transform_img(self, image):
        return self.to_tensor(image=image)


##모든 코드는 이 줄 위에 써주세요
_transform_entropoints = {
    'TransUnet': transform_transunet,
    'augmix' : transform_augmix,
    'copy_paste' : transform_copypaste
}


def transform_entrypoint(criterion_name):
    return _transform_entropoints[criterion_name]


def is_transform(criterion_name):
    return criterion_name in _transform_entropoints


def create_transforms(criterion_name, seed, **kwargs):
    if is_transform(criterion_name):
        create_fn = transform_entrypoint(criterion_name)
        criterion = create_fn(seed, **kwargs)
    else:
        raise RuntimeError('Unknown transform (%s)' % criterion_name)
    return criterion