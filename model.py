import torch
import torch.nn as nn
from torchvision import models
import segmentation_models_pytorch as smp
from ocrnet import HRNet, HRNet_Mscale


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
    def __init__(self, num_classes=11, pretrained=True):
        super().__init__()
        model = HRNet_Mscale(num_classes=num_classes)
        if pretrained:
            checkpoint = torch.load('/opt/ml/segmentation/semantic-segmentation-level2-cv-06/models/weights/cityscapes_trainval_ocr.HRNet_Mscale_nimble-chihuahua.pth')
            for k in list(checkpoint['state_dict'].keys()):
                name = k.replace('module.', '')
                checkpoint['state_dict'][name] = checkpoint['state_dict'].pop(k)
            
            model_state = model.state_dict()
            pretrained_state = checkpoint['state_dict']
            model_state = {k: v for k, v in model_state.items() if k in pretrained_state}
            pretrained_state.update(model_state)
            model.load_state_dict(pretrained_state)
            print('All keys matched successfully!!')
        self.model = model

    def forward(self, x):
        return self.model(x)

