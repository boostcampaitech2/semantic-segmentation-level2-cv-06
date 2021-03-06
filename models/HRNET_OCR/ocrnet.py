import torch
import torch.nn as nn
from models.HRNET_OCR.ocrnet_utils import (
    BNReLU,
    ResizeX,
    SpatialGather_Module,
    SpatialOCR_Module,
    Upsample,
    fmt_scale,
    get_trunk,
    initialize_weights,
    make_attn_head,
    scale_as,
)

INIT_DECODER = False
MID_CHANNELS = 512
KEY_CHANNELS = 256

MSCALE_LO_SCALE = 0.5
N_SCALES = [0.5, 1.0, 2.0]


class OCR_block(nn.Module):
    """
    Some of the code in this class is borrowed from:
    https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR
    """

    def __init__(self, high_level_ch, num_classes):
        super(OCR_block, self).__init__()

        ocr_mid_channels = MID_CHANNELS
        ocr_key_channels = KEY_CHANNELS
        num_classes = num_classes

        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(
                high_level_ch, ocr_mid_channels, kernel_size=3, stride=1, padding=1
            ),
            BNReLU(ocr_mid_channels),
        )
        self.ocr_gather_head = SpatialGather_Module(num_classes)
        self.ocr_distri_head = SpatialOCR_Module(
            in_channels=ocr_mid_channels,
            key_channels=ocr_key_channels,
            out_channels=ocr_mid_channels,
            scale=1,
            dropout=0.05,
        )
        self.cls_head = nn.Conv2d(
            ocr_mid_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True
        )

        self.aux_head = nn.Sequential(
            nn.Conv2d(high_level_ch, high_level_ch, kernel_size=1, stride=1, padding=0),
            BNReLU(high_level_ch),
            nn.Conv2d(
                high_level_ch,
                num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

        if INIT_DECODER:
            initialize_weights(
                self.conv3x3_ocr,
                self.ocr_gather_head,
                self.ocr_distri_head,
                self.cls_head,
                self.aux_head,
            )

    def forward(self, high_level_features):
        feats = self.conv3x3_ocr(high_level_features)
        aux_out = self.aux_head(high_level_features)
        context = self.ocr_gather_head(feats, aux_out)
        ocr_feats = self.ocr_distri_head(feats, context)
        cls_out = self.cls_head(ocr_feats)
        return cls_out, aux_out, ocr_feats


class OCRNet(nn.Module):
    """
    OCR net
    """

    def __init__(self, num_classes, trunk="hrnetv2"):
        super(OCRNet, self).__init__()
        self.backbone, _, _, high_level_ch = get_trunk(trunk)
        self.ocr = OCR_block(high_level_ch, num_classes)

    def forward(self, inputs):
        x = inputs

        _, _, high_level_features = self.backbone(x)
        cls_out, aux_out, _ = self.ocr(high_level_features)
        aux_out = scale_as(aux_out, x)
        cls_out = scale_as(cls_out, x)

        output_dict = {"pred": cls_out, "aux": aux_out}
        return output_dict


class MscaleOCR(nn.Module):
    """
    Mscale OCR net
    """

    def __init__(self, num_classes, trunk="hrnetv2"):
        super(MscaleOCR, self).__init__()
        self.backbone, _, _, high_level_ch = get_trunk(trunk)
        self.ocr = OCR_block(high_level_ch, num_classes)
        self.scale_attn = make_attn_head(in_ch=MID_CHANNELS, out_ch=1)

    def _fwd(self, x):
        x_size = x.size()[2:]

        _, _, high_level_features = self.backbone(x)
        cls_out, aux_out, ocr_mid_feats = self.ocr(high_level_features)
        attn = self.scale_attn(ocr_mid_feats)

        aux_out = Upsample(aux_out, x_size)
        cls_out = Upsample(cls_out, x_size)
        attn = Upsample(attn, x_size)

        return {"cls_out": cls_out, "aux_out": aux_out, "logit_attn": attn}

    def nscale_forward(self, inputs, scales):
        """
        Hierarchical attention, primarily used for getting best inference
        results.

        We use attention at multiple scales, giving priority to the lower
        resolutions. For example, if we have 4 scales {0.5, 1.0, 1.5, 2.0},
        then evaluation is done as follows:

              p_joint = attn_1.5 * p_1.5 + (1 - attn_1.5) * down(p_2.0)
              p_joint = attn_1.0 * p_1.0 + (1 - attn_1.0) * down(p_joint)
              p_joint = up(attn_0.5 * p_0.5) * (1 - up(attn_0.5)) * p_joint

        The target scale is always 1.0, and 1.0 is expected to be part of the
        list of scales. When predictions are done at greater than 1.0 scale,
        the predictions are downsampled before combining with the next lower
        scale.

        Inputs:
          scales - a list of scales to evaluate
          inputs - dict containing 'images', the input, and 'gts', the ground
                   truth mask

        Output:
          If training, return loss, else return prediction + attention
        """
        x_1x = inputs

        assert 1.0 in scales, "expected 1.0 to be the target scale"
        # Lower resolution provides attention for higher rez predictions,
        # so we evaluate in order: high to low
        scales = sorted(scales, reverse=True)

        pred = None
        aux = None
        output_dict = {}

        for s in scales:
            x = ResizeX(x_1x, s)
            outs = self._fwd(x)
            cls_out = outs["cls_out"]
            attn_out = outs["logit_attn"]
            aux_out = outs["aux_out"]

            output_dict[fmt_scale("pred", s)] = cls_out
            if s != 2.0:
                output_dict[fmt_scale("attn", s)] = attn_out

            if pred is None:
                pred = cls_out
                aux = aux_out
            elif s >= 1.0:
                # downscale previous
                pred = scale_as(pred, cls_out)
                pred = attn_out * cls_out + (1 - attn_out) * pred
                aux = scale_as(aux, cls_out)
                aux = attn_out * aux_out + (1 - attn_out) * aux
            else:
                # s < 1.0: upscale current
                cls_out = attn_out * cls_out
                aux_out = attn_out * aux_out

                cls_out = scale_as(cls_out, pred)
                aux_out = scale_as(aux_out, pred)
                attn_out = scale_as(attn_out, pred)

                pred = cls_out + (1 - attn_out) * pred
                aux = aux_out + (1 - attn_out) * aux

        output_dict["pred"] = pred
        output_dict["aux"] = aux
        return output_dict

    def two_scale_forward(self, inputs):
        """
        Do we supervised both aux outputs, lo and high scale?
        Should attention be used to combine the aux output?
        Normally we only supervise the combined 1x output

        If we use attention to combine the aux outputs, then
        we can use normal weighting for aux vs. cls outputs
        """
        x_1x = inputs

        x_lo = ResizeX(x_1x, MSCALE_LO_SCALE)
        lo_outs = self._fwd(x_lo)
        pred_05x = lo_outs["cls_out"]
        p_lo = pred_05x
        aux_lo = lo_outs["aux_out"]
        logit_attn = lo_outs["logit_attn"]
        attn_05x = logit_attn

        hi_outs = self._fwd(x_1x)
        pred_10x = hi_outs["cls_out"]
        p_1x = pred_10x
        aux_1x = hi_outs["aux_out"]

        p_lo = logit_attn * p_lo
        aux_lo = logit_attn * aux_lo
        p_lo = scale_as(p_lo, p_1x)
        aux_lo = scale_as(aux_lo, p_1x)

        logit_attn = scale_as(logit_attn, p_1x)

        # combine lo and hi predictions with attention
        joint_pred = p_lo + (1 - logit_attn) * p_1x
        joint_aux = aux_lo + (1 - logit_attn) * aux_1x

        output_dict = {
            "pred": joint_pred,
            "pred_05x": pred_05x,
            "pred_10x": pred_10x,
            "attn_05x": attn_05x,
            "aux": joint_aux,
        }
        return output_dict

    def forward(self, inputs):

        if N_SCALES and not self.training:
            return self.nscale_forward(inputs, N_SCALES)

        return self.two_scale_forward(inputs)


def HRNet(num_classes):
    return OCRNet(num_classes, trunk="hrnetv2")


def HRNet_Mscale(num_classes):
    return MscaleOCR(num_classes, trunk="hrnetv2")
