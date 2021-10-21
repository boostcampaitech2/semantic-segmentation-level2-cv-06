import torch.nn as nn
from torchvision import models
import segmentation_models_pytorch as smp
from hrnet.ocrnet import HRNet, HRNet_Mscale

class hrnet_ocr(nn.Module):
    def __init__(self, num_classes, pretrained = True, scale = 'single'):
        super().__init__()
        criterion = None
        if scale == 'single':
            self.model = HRNet(num_classes, criterion)
        else:
            self.model = HRNet_Mscale(num_classes, criterion)

    def forward(self, x):
        return self.model(x)


class FCNRes50(nn.Module):
    def __init__(self, num_classes=11, pretrained=True):
        super().__init__()
        self.model = models.segmentation.fcn_resnet50(pretrained=pretrained)
        
        # output class를 data set에 맞도록 수정
        self.model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.model(x)
        return x


class FCNRes101(nn.Module):
    def __init__(self, num_classes=11, pretrained=True):
        super().__init__()
        self.model = models.segmentation.fcn_resnet101(pretrained=pretrained)
        
        # output class를 data set에 맞도록 수정
        self.model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.model(x)
        return x


class DeepLabV3_Res50(nn.Module):
    def __init__(self, num_classes=11, pretrained=True):
        super().__init__()
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained)
        
        # output class를 data set에 맞도록 수정
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.model(x)
        return x


class DeepLabV3_Res101(nn.Module):
    def __init__(self, num_classes=11, pretrained=True):
        super().__init__()
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained)
        
        # output class를 data set에 맞도록 수정
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.model(x)
        return x


class UNet(nn.Module):
    def __init__(self, encoder='efficientnet-b0', weights='imagenet', num_classes=11, pretrained=True):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder,       # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=weights,    # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,              # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_classes,        # model output channels (number of classes in your dataset)
            )
    
    def forward(self, x):
        x = self.model(x)
        return x


class UNet_PlusPlus(nn.Module):
    def __init__(self, encoder='efficientnet-b0', weights='imagenet', num_classes=11, pretrained=True):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder,       # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=weights,    # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,              # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_classes,        # model output channels (number of classes in your dataset)
            )
    
    def forward(self, x):
        x = self.model(x)
        return x


class FPN(nn.Module):
    def __init__(self, encoder='efficientnet-b0', weights='imagenet', num_classes=11, pretrained=True):
        super().__init__()
        self.model = smp.FPN(
            encoder_name=encoder,       # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=weights,    # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,              # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_classes,        # model output channels (number of classes in your dataset)
            )
    
    def forward(self, x):
        x = self.model(x)
        return x


class PSPNet(nn.Module):
    def __init__(self, encoder='efficientnet-b0', weights='imagenet', num_classes=11, pretrained=True):
        super().__init__()
        self.model = smp.PSPNet(
            encoder_name=encoder,       # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=weights,    # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,              # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_classes,        # model output channels (number of classes in your dataset)
            )
    
    def forward(self, x):
        x = self.model(x)
        return x


class DeepLabV3(nn.Module):
    def __init__(self, encoder='efficientnet-b0', weights='imagenet', num_classes=11, pretrained=True):
        super().__init__()
        self.model = smp.DeepLabV3(
            encoder_name=encoder,       # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=weights,    # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,              # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_classes,        # model output channels (number of classes in your dataset)
            )
    
    def forward(self, x):
        x = self.model(x)
        return x


class DeepLabV3_Plus(nn.Module):
    def __init__(self, encoder='efficientnet-b0', weights='imagenet', num_classes=11, pretrained=True):
        super().__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder,       # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=weights,    # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,              # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_classes,        # model output channels (number of classes in your dataset)
            )
    
    def forward(self, x):
        x = self.model(x)
        return x
