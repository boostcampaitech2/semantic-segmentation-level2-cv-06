import torch
import torch.nn as nn
import torch.nn.functional as F

import loss.rmi_utils as rmi_utils

_CLIP_MIN = 1e-6  # min clip value after softmax or sigmoid operations
_POS_ALPHA = 5e-4  # add this factor to ensure the AA^T is positive definite
_IS_SUM = 1  # sum the loss per channel

NUM_OUTPUTS = 1
ALIGN_CORNERS = True
BALANCE_WEIGHTS = [1]


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=11, smoothing=0.2, dim=-1, weight=None):
        """if smoothing == 0, it's one-hot method
        if 0 < smoothing < 1, it's smooth method
        """
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target, do_rmi=False):
        # do_rmi: dummy var
        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=self.dim)

        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class RMILoss(nn.Module):
    """
    region mutual information
    I(A, B) = H(A) + H(B) - H(A, B)
    This version need a lot of memory if do not dwonsample.
    """

    def __init__(
        self,
        num_classes=11,
        rmi_radius=3,
        rmi_pool_way=1,
        rmi_pool_size=4,
        rmi_pool_stride=4,
        loss_weight_lambda=0.5,
        lambda_way=1,
        ignore_index=255,
    ):
        super(RMILoss, self).__init__()
        self.num_classes = num_classes
        # radius choices
        assert rmi_radius in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.rmi_radius = rmi_radius
        assert rmi_pool_way in [0, 1, 2, 3]
        self.rmi_pool_way = rmi_pool_way

        # set the pool_size = rmi_pool_stride
        assert rmi_pool_size == rmi_pool_stride
        self.rmi_pool_size = rmi_pool_size
        self.rmi_pool_stride = rmi_pool_stride
        self.weight_lambda = loss_weight_lambda
        self.lambda_way = lambda_way

        # dimension of the distribution
        self.half_d = self.rmi_radius * self.rmi_radius
        self.d = 2 * self.half_d
        self.kernel_padding = self.rmi_pool_size // 2
        # ignore class
        self.ignore_index = ignore_index

    def forward(self, logits_4D, labels_4D, do_rmi=True):
        # explicitly disable fp16 mode because torch.cholesky and
        # torch.inverse aren't supported by half
        logits_4D.float()
        labels_4D.float()
        # if cfg.TRAIN.FP16:
        #     with amp.disable_casts():
        #         loss = self.forward_sigmoid(logits_4D, labels_4D, do_rmi=do_rmi)
        # else:
        loss = self.forward_sigmoid(logits_4D, labels_4D, do_rmi=do_rmi)
        return loss

    def forward_sigmoid(self, logits_4D, labels_4D, do_rmi=False):
        """
        Using the sigmiod operation both.
        Args:
                logits_4D 	:	[N, C, H, W], dtype=float32
                labels_4D 	:	[N, H, W], dtype=long
                do_rmi          :       bool
        """
        # label mask -- [N, H, W, 1]
        label_mask_3D = labels_4D < self.num_classes

        # valid label
        valid_onehot_labels_4D = F.one_hot(
            labels_4D.long() * label_mask_3D.long(), num_classes=self.num_classes
        ).float()
        label_mask_3D = label_mask_3D.float()
        label_mask_flat = label_mask_3D.view(
            [
                -1,
            ]
        )
        valid_onehot_labels_4D = valid_onehot_labels_4D * label_mask_3D.unsqueeze(dim=3)
        valid_onehot_labels_4D.requires_grad_(False)

        # PART I -- calculate the sigmoid binary cross entropy loss
        valid_onehot_label_flat = valid_onehot_labels_4D.view(
            [-1, self.num_classes]
        ).requires_grad_(False)
        logits_flat = (
            logits_4D.permute(0, 2, 3, 1).contiguous().view([-1, self.num_classes])
        )

        # binary loss, multiplied by the not_ignore_mask
        valid_pixels = torch.sum(label_mask_flat)
        binary_loss = F.binary_cross_entropy_with_logits(
            logits_flat,
            target=valid_onehot_label_flat,
            weight=label_mask_flat.unsqueeze(dim=1),
            reduction="sum",
        )
        bce_loss = torch.div(binary_loss, valid_pixels + 1.0)
        if not do_rmi:
            return bce_loss

        # PART II -- get rmi loss
        # onehot_labels_4D -- [N, C, H, W]
        probs_4D = logits_4D.sigmoid() * label_mask_3D.unsqueeze(dim=1) + _CLIP_MIN
        valid_onehot_labels_4D = valid_onehot_labels_4D.permute(
            0, 3, 1, 2
        ).requires_grad_(False)

        # get region mutual information
        rmi_loss = self.rmi_lower_bound(valid_onehot_labels_4D, probs_4D)

        # add together
        # logx.msg(f'lambda_way {self.lambda_way}')
        # logx.msg(f'bce_loss {bce_loss} weight_lambda {self.weight_lambda} rmi_loss {rmi_loss}')
        if self.lambda_way:
            final_loss = self.weight_lambda * bce_loss + rmi_loss * (
                1 - self.weight_lambda
            )
        else:
            final_loss = bce_loss + rmi_loss * self.weight_lambda

        return final_loss

    def inverse(self, x):
        return torch.inverse(x)

    def rmi_lower_bound(self, labels_4D, probs_4D):
        """
        calculate the lower bound of the region mutual information.
        Args:
                labels_4D 	:	[N, C, H, W], dtype=float32
                probs_4D 	:	[N, C, H, W], dtype=float32
        """
        assert labels_4D.size() == probs_4D.size()

        p, s = self.rmi_pool_size, self.rmi_pool_stride
        if self.rmi_pool_stride > 1:
            if self.rmi_pool_way == 0:
                labels_4D = F.max_pool2d(
                    labels_4D, kernel_size=p, stride=s, padding=self.kernel_padding
                )
                probs_4D = F.max_pool2d(
                    probs_4D, kernel_size=p, stride=s, padding=self.kernel_padding
                )
            elif self.rmi_pool_way == 1:
                labels_4D = F.avg_pool2d(
                    labels_4D, kernel_size=p, stride=s, padding=self.kernel_padding
                )
                probs_4D = F.avg_pool2d(
                    probs_4D, kernel_size=p, stride=s, padding=self.kernel_padding
                )
            elif self.rmi_pool_way == 2:
                # interpolation
                shape = labels_4D.size()
                new_h, new_w = shape[2] // s, shape[3] // s
                labels_4D = F.interpolate(
                    labels_4D, size=(new_h, new_w), mode="nearest"
                )
                probs_4D = F.interpolate(
                    probs_4D, size=(new_h, new_w), mode="bilinear", align_corners=True
                )
            else:
                raise NotImplementedError("Pool way of RMI is not defined!")
        # we do not need the gradient of label.
        label_shape = labels_4D.size()
        n, c = label_shape[0], label_shape[1]

        # combine the high dimension points from label and probability map. new shape [N, C, radius * radius, H, W]
        la_vectors, pr_vectors = rmi_utils.map_get_pairs(
            labels_4D, probs_4D, radius=self.rmi_radius, is_combine=0
        )

        la_vectors = (
            la_vectors.view([n, c, self.half_d, -1])
            .type(torch.cuda.DoubleTensor)
            .requires_grad_(False)
        )
        pr_vectors = pr_vectors.view([n, c, self.half_d, -1]).type(
            torch.cuda.DoubleTensor
        )

        # small diagonal matrix, shape = [1, 1, radius * radius, radius * radius]
        diag_matrix = torch.eye(self.half_d).unsqueeze(dim=0).unsqueeze(dim=0)

        # the mean and covariance of these high dimension points
        # Var(X) = E(X^2) - E(X) E(X), N * Var(X) = X^2 - X E(X)
        la_vectors = la_vectors - la_vectors.mean(dim=3, keepdim=True)
        la_cov = torch.matmul(la_vectors, la_vectors.transpose(2, 3))

        pr_vectors = pr_vectors - pr_vectors.mean(dim=3, keepdim=True)
        pr_cov = torch.matmul(pr_vectors, pr_vectors.transpose(2, 3))
        # https://github.com/pytorch/pytorch/issues/7500
        # waiting for batched torch.cholesky_inverse()
        # pr_cov_inv = torch.inverse(pr_cov + diag_matrix.type_as(pr_cov) * _POS_ALPHA)
        pr_cov_inv = self.inverse(pr_cov + diag_matrix.type_as(pr_cov) * _POS_ALPHA)
        # if the dimension of the point is less than 9, you can use the below function
        # to acceleration computational speed.
        # pr_cov_inv = utils.batch_cholesky_inverse(pr_cov + diag_matrix.type_as(pr_cov) * _POS_ALPHA)

        la_pr_cov = torch.matmul(la_vectors, pr_vectors.transpose(2, 3))
        # the approxiamation of the variance, det(c A) = c^n det(A), A is in n x n shape;
        # then log det(c A) = n log(c) + log det(A).
        # appro_var = appro_var / n_points, we do not divide the appro_var by number of points here,
        # and the purpose is to avoid underflow issue.
        # If A = A^T, A^-1 = (A^-1)^T.
        appro_var = la_cov - torch.matmul(
            la_pr_cov.matmul(pr_cov_inv), la_pr_cov.transpose(-2, -1)
        )
        # appro_var = la_cov - torch.chain_matmul(la_pr_cov, pr_cov_inv, la_pr_cov.transpose(-2, -1))
        # appro_var = torch.div(appro_var, n_points.type_as(appro_var)) + diag_matrix.type_as(appro_var) * 1e-6

        # The lower bound. If A is nonsingular, ln( det(A) ) = Tr( ln(A) ).
        rmi_now = 0.5 * rmi_utils.log_det_by_cholesky(
            appro_var + diag_matrix.type_as(appro_var) * _POS_ALPHA
        )
        # rmi_now = 0.5 * torch.logdet(appro_var + diag_matrix.type_as(appro_var) * _POS_ALPHA)

        # mean over N samples. sum over classes.
        rmi_per_class = rmi_now.view([-1, self.num_classes]).mean(dim=0).float()
        # is_half = False
        # if is_half:
        # 	rmi_per_class = torch.div(rmi_per_class, float(self.half_d / 2.0))
        # else:
        rmi_per_class = torch.div(rmi_per_class, float(self.half_d))

        rmi_loss = torch.sum(rmi_per_class) if _IS_SUM else torch.mean(rmi_per_class)
        return rmi_loss


# https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction,
        )


# https://github.com/Beckschen/TransUNet/blob/d68a53a2da73ecb496bb7585340eb660ecda1d59/utils.py#L9
class DiceLoss(nn.Module):
    def __init__(self, n_classes=11):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(
        self,
        inputs,
        target,
        weight=[0.5, 1.5, 1, 2, 2, 1, 2, 1, 1, 3, 2],
        softmax=True,
        do_rmi=False,
    ):
        # do_rmi: dummy var for compatibility
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert (
            inputs.size() == target.size()
        ), "predict {} & target {} shape do not match".format(
            inputs.size(), target.size()
        )
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class OhemCrossEntropy(nn.Module):
    def __init__(
        self,
        ignore_label=-1,
        thres=0.7,
        min_kept=100000,
        weight=torch.FloatTensor(
            [
                1.0,
                2.2438960966453663,
                1.0367368849500223,
                3.684620397193553,
                3.844600329110712,
                3.7614326467956447,
                2.138889000323256,
                2.972463646614637,
                1.2325602611778486,
                6.039683119378457,
                5.002939087461785,
            ]
        ).cuda(),
    ):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_label, reduction="none"
        )

    def _ce_forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(
                input=score, size=(h, w), mode="bilinear", align_corners=ALIGN_CORNERS
            )

        loss = self.criterion(score, target)

        return loss

    def _ohem_forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(
                input=score, size=(h, w), mode="bilinear", align_corners=ALIGN_CORNERS
            )
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = (
            pred.contiguous()
            .view(
                -1,
            )[mask]
            .contiguous()
            .sort()
        )
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target):

        if NUM_OUTPUTS == 1:
            score = [score]

        weights = BALANCE_WEIGHTS
        assert len(weights) == len(score)

        functions = [self._ce_forward] * (len(weights) - 1) + [self._ohem_forward]
        return sum(
            [w * func(x, target) for (w, x, func) in zip(weights, score, functions)]
        )


_criterion_entropoints = {
    "cross_entropy": nn.CrossEntropyLoss,
    "focal": FocalLoss,
    "rmi": RMILoss,
    "smooth": LabelSmoothingLoss,
    "dice": DiceLoss,
    "ohem_cross_entropy": OhemCrossEntropy,
}


def criterion_entrypoint(criterion_name):
    return _criterion_entropoints[criterion_name]


def is_criterion(criterion_name):
    return criterion_name in _criterion_entropoints


def create_criterion(criterion_name, **kwargs):
    if is_criterion(criterion_name):
        create_fn = criterion_entrypoint(criterion_name)
        criterion = create_fn(**kwargs)
    else:
        raise RuntimeError("Unknown loss (%s)" % criterion_name)
    return criterion
