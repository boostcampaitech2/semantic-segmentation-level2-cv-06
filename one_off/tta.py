import ttach as tta
class custom_tta():
    def __init__(self, key):
        self.transform = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Rotate90(angles=[0, 180]),
                tta.Scale(scales=[0.5, 1, 2])   
            ])
        self.key = key
    
    def get_tta(self, model):
        return tta.SegmentationTTAWrapper(model, self.transform, output_mask_key=self.key)