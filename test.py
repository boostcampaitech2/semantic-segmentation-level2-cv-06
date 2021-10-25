import torch
from ocrnet import HRNet_Mscale


model = HRNet_Mscale(num_classes=11)
print(model.state_dict()['backbone.stage4.2.fuse_layers.3.2.0.1.weight'])

checkpoint = torch.load('/opt/ml/segmentation/semantic-segmentation-level2-cv-06/models/weights/cityscapes_trainval_ocr.HRNet_Mscale_nimble-chihuahua.pth')
# checkpoint = torch.load('/opt/ml/segmentation/semantic-segmentation-level2-cv-06/models/weights/cityscapes_ocrnet.HRNet_Mscale_outstanding-turtle.pth')            

for k in list(checkpoint['state_dict'].keys()):
    name = k.replace('module.', '')
    checkpoint['state_dict'][name] = checkpoint['state_dict'].pop(k)

print(checkpoint['state_dict']['backbone.stage4.2.fuse_layers.3.2.0.1.weight'])

