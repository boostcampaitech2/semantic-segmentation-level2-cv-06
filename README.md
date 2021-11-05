# ✨칠성사이다✨
<img src="https://user-images.githubusercontent.com/20790778/137433985-622be56d-82eb-4dd7-bbec-c7079b0bf059.png" width=700 height=393 />

| [강지우](https://github.com/jiwoo0212) | [곽지윤](https://github.com/kwakjeeyoon) | [서지유](https://github.com/JiyouSeo) | [송나은](https://github.com/sne12345) | [오재환](https://github.com/jaehwan-AI) | [이준혁](https://github.com/kmouleejunhyuk) | [전경재](https://github.com/ppskj178) |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| ![image](https://user-images.githubusercontent.com/68782183/138297784-223d2d61-74f7-4a19-8aaf-5525309e2bd8.jpg) | ![image](https://user-images.githubusercontent.com/55044675/138575690-216fc641-dba1-4737-a571-6b1058e780b2.jpg) | ![image](https://avatars.githubusercontent.com/u/61641072?v=4) | ![image](https://user-images.githubusercontent.com/68782183/138638320-19b24d42-6014-4042-b443-cbeb50251cfd.jpg) | ![image](https://user-images.githubusercontent.com/68782183/138295480-ca0169cd-5c40-44ae-b222-d74d9cc4bc82.jpg) | ![d](https://user-images.githubusercontent.com/49234207/138424590-385b34c2-fae2-426f-8abe-8b40d21ba766.jpg)| ![image](https://user-images.githubusercontent.com/20790778/138396418-b669cbed-40b0-45eb-9f60-7167cae739b7.png) | |


## Competition Overview
- 목적: 사진에서 쓰레기를 Segmentation 하는 모델 제작
- Dataset: 일반 쓰레기, 플라스틱, 종이, 유리 등 11 종류의 쓰레기가 찍힌 사진 데이터셋 4091여장(train 2617장, valid : 665장, test : 819장)
- 평가 metric: mean Intersection over Union(mIOU) on test dataset <br /><br />
![segmentation_viz](https://user-images.githubusercontent.com/51853700/140501597-8192b523-2152-4734-9a78-d474321ac6d2.png)


## Project Roadmap
<!-- ![0001](https://user-images.githubusercontent.com/68782183/140489733-83112ac8-65bb-4fa7-b4d1-dbf1198583d3.jpg) -->
<img src="https://user-images.githubusercontent.com/68782183/140489733-83112ac8-65bb-4fa7-b4d1-dbf1198583d3.jpg" width=700 height=393 />

## contents
```
|-- datasets
|   |-- coco.py
|   |-- copy_paste.py
|   |-- dataset.py
|   `-- transform_test.py
|-- loss
|   |-- losses.py
|   `-- rmi_utils.py
|-- models
|   |-- HRNET_OCR
|   |   |-- hrnetv2.py
|   |   |-- ocrnet.py
|   |   `-- ocrnet_utils.py
|   |-- TransUnet
|   |   |-- vit_seg_configs.py
|   |   |-- vit_seg_modeling.py
|   |   `-- vit_seg_modeling_resnet_skip.py
|   `-- model.py
|-- optimizer
|   |-- optim_sche.py
|   `-- radam.py
|-- sample_data
|   |-- image
|   `-- train.json
|-- utils
|    |-- densecrf.py
|    |-- ensemble.ipynb
|    |-- img_diff.py
|    |-- labelcount.py
|    |-- new_copy_paste
|    |   |-- new_copy_paste.py
|    |   `-- new_copy_paste_dataset.py
|    `-- utils.py
|-- train.py
|-- inference.py
|-- class_dict.csv
|-- README.md
```


## best result
- ./runs/Transunet_SGD_1024.pt
  - Public : 0.672, private : 0.642
  - **TransUNet -** [weight download](https://drive.google.com/drive/folders/1TlLYkIUscPMfkEd6Oy4zgef71-WbeSx6)
- ./runs/OCRNet_augmix.pt
  - Public : 0.579, private : 0.52
  - **OCRNet + Augmix -** [weight download](https://drive.google.com/drive/folders/1Ouy1AaO4ZVQ1IvMJJwhJP7XM_aVq_w5-)
- ./runs/DeepLabv3_efficientb7_copypaste.pt
  - Public : 0.599, private : 0.583
  - **DeepLabV3 + copypaste -** [weight download](https://drive.google.com/drive/folders/1z3ribIiZ8on-v624r2j0TN5ihx4M67ro)
- Ensemble (TransUNet+DeepLabV3)
  - Public : 0.707, private : 0.661

## simple start

### environment
```python
pip install requirement.txt
```
- [pydensecrf](https://github.com/lucasb-eyer/pydensecrf)

```python
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

### Train
```python
python train.py --model MscaleOCRNet --batch_size 10 --wandb True --custom_trs True
                --model DeepLabV3
                --model TransUnet
```  

### Inference
```python
python inference.py
```  
### ensemble
for ensemble, please reffer to `ensemble.ipynb`


## reference
[Hierarchical Multi-Scale Attention for Semantic Segmentation](https://github.com/NVIDIA/semantic-segmentation)

[Copy Paste](https://github.com/conradry/copy-paste-aug)

[TransUNet](https://github.com/Beckschen/TransUNet)

