import albumentations as A
import numpy as np
import pandas as pd
import pydensecrf.densecrf as dcrf
import ray
from pydensecrf.utils import unary_from_labels
from skimage.io import imread
from tqdm import tqdm

OUTPUT_CSV = "./output/transunet.csv"
IMAGE_PATH = "./input/data/"
OUTPUT_PATH = "./output/"

POS_W = 3
POS_XY_STD = 1
Bi_W = 4
Bi_XY_STD = 67
Bi_RGB_STD = 3


@ray.remote
def decode(rle_mask):
    mask = rle_mask.split()
    img = np.zeros(256 * 256, dtype=np.uint8)
    for (
        i,
        m,
    ) in enumerate(mask):
        img[i] = int(m)
    return img.reshape(256, 256)


@ray.remote
def encode(im):
    pixels = im.flatten()
    return " ".join(str(x) for x in pixels)


@ray.remote
def crf(original_image, mask_img):
    labels = mask_img.flatten()

    # Setting up the CRF model
    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], 11)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, 11, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(
        sxy=POS_XY_STD,
        compat=POS_W,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(
        sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=original_image.copy(), compat=Bi_W
    )

    Q = d.inference(50)  # MAX_ITER

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    return MAP.reshape((original_image.shape[0], original_image.shape[1]))


@ray.remote
def read_img(img_path):
    img = imread(img_path)
    transformed = transform(image=img)
    del img
    orig_img = transformed["image"]
    return orig_img


ray.init(ignore_reinit_error=True)
df = pd.read_csv(OUTPUT_CSV)
transform = A.Compose([A.Resize(256, 256)])

orig_img = [read_img.remote(IMAGE_PATH + df.loc[i, "image_id"]) for i in range(len(df))]
decoded_mask = [decode.remote(df.loc[i, "PredictionString"]) for i in range(len(df))]

orig_imgs = ray.get(orig_img)
decoded_masks = ray.get(decoded_mask)

crf_output = [crf.remote(orig_imgs[i], decoded_masks[i]) for i in range(len(df))]
crf_outputs = ray.get(crf_output)

encode_output = [encode.remote(crf_outputs[i]) for i in range(len(df))]
encode_outputs = ray.get(encode_output)

for i in tqdm(range(len(df))):
    df.loc[i, "PredictionString"] = encode_outputs[i]

df.to_csv(
    OUTPUT_PATH + f"crf{POS_W}_{POS_XY_STD}_{Bi_W}_{Bi_XY_STD}_{Bi_RGB_STD}.csv",
    index=False,
)

ray.shutdown()
