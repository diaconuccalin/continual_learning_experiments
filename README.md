# Continual Learning Experiments

CORe50 data obtained from [here](https://vlomonaco.github.io/core50/index.html#download). The following directory structure is needed, relative to project root directory, for the project to work properly:

~~~
.
├── datasets
│   ├── core50
│   │   ├── data
│   │   │   ├── core50_128x128
│   │   │   │   ├── [data unzipped from downloaded archive]
│   │   │   │   ...
│   │   │   ├── core50_350x350
│   │   │   │   ├── [data unzipped from downloaded archive]
│   │   │   │   ...
│   │   │   ├── labels.pkl
│   │   │   ├── LUP.pkl
│   │   │   ├── paths.pkl
...
~~~

Imagenet test images obtained from [here](https://www.kaggle.com/c/imagenet-object-localization-challenge/data).

ViT weights, pretrained on Imagenet 1k, obtained from here: [B_16](https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth) and [B_32](https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_32_imagenet1k.pth). The following directory structure is needed, relative to project root directory, for the project to work properly:

~~~
.
├── weights
│   ├── pretrained_imagenet
│   │   ├──B_16_imagenet1k.pth
│   │   ├── B_32_imagenet1k.pth
...
~~~
