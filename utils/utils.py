# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class_colormap = pd.read_csv(
    "/opt/ml/segmentation/semantic-segmentation-level2-cv-06/class_dict.csv"
)


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def label_accuracy_score(hist):
    """
    Returns accuracy score evaluation result.
      - [acc]: overall accuracy
      - [acc_cls]: mean accuracy
      - [mean_iu]: mean IU
      - [fwavacc]: fwavacc
    """
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    with np.errstate(divide="ignore", invalid="ignore"):
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)

    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc, iu


def add_hist(hist, label_trues, label_preds, n_class):
    """
    stack hist(confusion matrix)
    """

    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)

    return hist


def grid_image(images, masks, preds, n=4, shuffle=False):
    batch_size = masks.shape[0]
    if n > batch_size:
        n = batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(
        figsize=(12, 16)
    )  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다.
    gs = figure.add_gridspec(4, 3)
    ax = [None for _ in range(12)]

    for idx, choice in enumerate(choices):
        image = images[choice]
        mask = masks[choice]
        pred = preds[choice]

        ax[idx * 3] = figure.add_subplot(gs[idx, 0])
        ax[idx * 3].imshow(image)
        ax[idx * 3].grid(False)

        ax[idx * 3 + 1] = figure.add_subplot(gs[idx, 1])
        ax[idx * 3 + 1].imshow(label_to_color_image(mask))
        ax[idx * 3 + 1].grid(False)

        ax[idx * 3 + 2] = figure.add_subplot(gs[idx, 2])
        ax[idx * 3 + 2].imshow(label_to_color_image(pred))
        ax[idx * 3 + 2].grid(False)
        # 나중에 확률 값으로 얼마나 틀렸는지 시각화 해주는 열을 추가하면 더 좋을듯?

    figure.suptitle("image / GT / pred", fontsize=16)

    return figure


def create_trash_label_colormap():
    """Creates a label colormap used in Trash segmentation.
    Returns:
        A colormap for visualizing segmentation results.
    """
    global class_colormap

    colormap = np.zeros((11, 3), dtype=np.uint8)
    for inex, (_, r, g, b) in enumerate(class_colormap.values):
        colormap[inex] = [r, g, b]

    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.
    Args:
        label: A 2D array with integer type, storing the segmentation label.
    Returns:
        result: A 2D array with floating type. The element of the array
                is the color indexed by the corresponding element in the input label
                to the trash color map.
    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
              map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError("Expect 2-D input label")

    colormap = create_trash_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError("label value too large.")

    return colormap[label]


def remove_dot_underbar(root):
    for path in os.listdir(root):
        tmp = os.path.join(root, path)
        if os.path.isdir(tmp):
            remove_dot_underbar(tmp)
        if path[0] == "." and path[1] == "_":
            os.remove(tmp)
            print(tmp, "is removed")


def is_battery(mask):
    if 9 in mask:
        return True
    else:
        return False
