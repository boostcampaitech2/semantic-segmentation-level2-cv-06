import torch
import torch.nn as nn
from torchvision import models
import segmentation_models_pytorch as smp
from models.ocrnet import HRNet, HRNet_Mscale


class FCNRes50(nn.Module):
    def __init__(self, num_classes=11, pretrained=True):
        super().__init__()
        self.model = models.segmentation.fcn_resnet50(pretrained=pretrained)
        
        # output class를 data set에 맞도록 수정
        self.model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)


class FCNRes101(nn.Module):
    def __init__(self, num_classes=11, pretrained=True):
        super().__init__()
        self.model = models.segmentation.fcn_resnet101(pretrained=pretrained)
        
        # output class를 data set에 맞도록 수정
        self.model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)


class DeepLabV3_Res50(nn.Module):
    def __init__(self, num_classes=11, pretrained=True):
        super().__init__()
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained)
        
        # output class를 data set에 맞도록 수정
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)


class DeepLabV3_Res101(nn.Module):
    def __init__(self, num_classes=11, pretrained=True):
        super().__init__()
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained)
        
        # output class를 data set에 맞도록 수정
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)


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
        return self.model(x)


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
        return self.model(x)


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
        return self.model(x)


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
        return self.model(x)


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
        return self.model(x)


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
        return self.model(x)


class OCRNet(nn.Module):
    def __init__(self, num_classes=11, pretrained=True):
        super().__init__()
        self.model = HRNet(num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)


class MscaleOCRNet(nn.Module):
    class module(nn.Module):
        def __init__(self, num_classes=11):
            super().__init__()
            self.module = HRNet_Mscale(num_classes=num_classes)

    def __init__(self, num_classes=11, pretrained=True):
        super().__init__()
        self.model = self.module(num_classes = num_classes)
        if pretrained:
            checkpoint = torch.load('/opt/ml/segmentation/semantic-segmentation-level2-cv-06/models/weights/cityscapes_ocrnet.HRNet_Mscale_outstanding-turtle.pth')
            checkpoint = checkpoint['state_dict']
            model_state_dict = self.model.state_dict()
            
            for k in model_state_dict.keys():
                if k not in checkpoint:
                    raise Exception("model state dict load key error")
                elif model_state_dict[k].size() == checkpoint[k].size():
                    model_state_dict[k] = checkpoint[k]
                else:
                    print(f"model state dict load skip {k}")

            self.model.load_state_dict(model_state_dict)


    def forward(self, x):
        return self.model(x)

