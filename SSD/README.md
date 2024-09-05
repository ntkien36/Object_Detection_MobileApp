# SSD300 Model with Pruning, Finetuning and QAT.
A [PyTorch](http://pytorch.org/) implementation of:
- [Single Shot MultiBox Detector](http://arxiv.org/abs/1512.02325) from the 2016 paper by Wei Liu, etc.
- Variants of SSD with Resnet-50/MobileNetv1/MobileNetv2 backbones.
- [Single-Shot Refinement Neural Network for Object Detection](https://arxiv.org/abs/1711.06897) from the 2017 paper by Shifeng Zhang, etc.
- [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710) from the 2017 paper by Hao Li, etc.
- Quantization Aware Training.
  

### List
- <a href='#installation'>Installation</a>
- <a href='#datasets'>Datasets</a>
- <a href='#train-and-test'>Train and Test</a>
- <a href='#evaluation'>Evaluate</a>
- <a href='#prune-and-finetune'>Prune and Finetune</a>
- <a href='#qat'>QAT</a>
- <a href='#demos'>Demos</a>
- <a href='#references'>Reference</a>

&nbsp;
&nbsp;
&nbsp;
&nbsp;

## Installation
- Clone this repository.
- Then download the dataset by following the [instructions](#datasets) below.

## Datasets
We provide bash scripts to handle the dataset downloads and setup.  We also provide simple dataset loaders that inherit `torch.utils.data.Dataset`, making them fully compatible with the `torchvision.datasets` [API](http://pytorch.org/docs/torchvision/datasets.html).

Please refer to `config.py` file (path: ssd.pytorch.tencent/data) and update dataset root if necessary. Please also note that dataset root for VALIDATION should be written within `config.py`, while dataset root for TRAINING can be updated through args during execution of a program. 

### VOC Dataset
PASCAL VOC: Visual Object Classes

##### Download VOC2007 trainval & test
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```

##### Download VOC2012 trainval
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```

```
### COCO (Not supported now but can refer to pycocotools API)
Microsoft COCO: Common Objects in Context

##### Download COCO 2014
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/COCO2014.sh
```

## Train and Test
### Model Set-up
- You can get all required backbone weights which have existed in `ssd.pytorch.tencent/weights` dir. 

```Shell
# navigate them by:
mkdir weights
cd weights
```
Based on this design, all backbone layers returned by functions in `backbones.py` have already stored pretrained weights.

### Training SSD (Resnet/VGG/MobileNetv1/MobileNetv2)
- To train SSD using the train script simply specify the parameters listed in `train_test_vrmSSD.py` as a flag or manually change them.

```Shell
#Use VOC dataset by default
#Train + Test SSD model with vgg backbone
python train_test_vrmSSD.py --evaluate True # testing while training
python train_test_vrmSSD.py # only training

#Train + Test SSD model with resnet backbone
python train_test_vrmSSD.py --use_res --evaluate True # testing while training
python train_test_vrmSSD.py --use_res # only training

#Train + Test SSD model with mobilev1 backbone
python train_test_vrmSSD.py --use_m1 --evaluate True # testing while training
python train_test_vrmSSD.py --use_m1 # only training

#Train + Test SSD model with mobilev2 backbone
python train_test_vrmSSD.py --use_m2 --evaluate True # testing while training
python train_test_vrmSSD.py --use_m2 # only training

```

- Note:
 
  * You can pick-up training from a checkpoint by specifying the path as one of the training parameters (again, see `--resume` for options)

## Evaluation
### Evaluate SSD (Resnet/VGG/MobileNetv1/MobileNetv2)
Use `eval_voc_vrmSSD.py` to evaluate.
```Shell
#Model evaluation on VOC for vggSSD separately
python eval_voc_vrmSSD.py --trained_model weights/_your_trained_SSD_model_.pth

#Model evaluation on VOC for resnetSSD separately
python eval_voc_vrmSSD.py --use_res --trained_model weights/_your_trained_SSD_model_.pth

#Model evaluation on VOC for mobileSSD v1 separately
python eval_voc_vrmSSD.py --use_m1 --trained_model weights/_your_trained_SSD_model_.pth

#Model evaluation on VOC for mobileSSD v2 separately
python eval_voc_vrmSSD.py --use_m2 --trained_model weights/_your_trained_SSD_model_.pth
```  
You can evaluate some scores from the `Eval.ipynb`.
## Prune and Finetune
### Prune
- `prune_weights_vggSSD.py`

```Shell
#Use absolute weights-based criterion for filter pruning on vggSSD
python prune_weights_vggSSD.py --trained_model weights/_your_trained_model_.pth

#**Note**
#Due to the limitation of PyTorch, if you really need to prune left path conv layer,
#after call this file, please use prune_rbconv_by_number() MANUALLY to prune all following right bottom layers affected by your pruning

```
The way of loading trained model (first time) and finetuned model (> 2 times) are different.
Please change the following codes within `rune_weights_vggSSD.py` file.
```Shell
# ------------------------------------------- 1st prune: load model from state_dict
model = build_ssd('train', cfg, cfg['min_dim'], cfg['num_classes'], base='resnet').cuda()
state_dict = torch.load(args.trained_model)
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    head = k[:7] # head = k[:4]
    if head == 'module.': # head == 'vgg.', module. is due to DataParellel
        name = k[7:]  # name = 'base.' + k[4:]
    else:
        name = k
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
#model.load_state_dict(torch.load(args.trained_model))

# ------------------------------------------- >= 2nd prune: load model from previous pruning
model = torch.load(args.trained_model).cuda()
```

### Finetune
- `finetune_vggresSSD.py`
```Shell
#Finetune prunned model vggSSD (Train/Test on VOC)
python finetune_vggresSSD.py --pruned_model prunes/_your_prunned_model_ --lr x --epoch y
```

## QAT
Following the code in `QAT.ipynb`.

## Demos

### Try the demo notebook
- Make sure you have [jupyter notebook](http://jupyter.readthedocs.io/en/latest/install.html) installed.
- Two alternatives for installing jupyter notebook:
    1. If you installed PyTorch with [conda](https://www.continuum.io/downloads) (recommended), then you should already have it.  (Just  navigate to the ssd.pytorch cloned repo and run):
    `jupyter notebook`

    2. If using [pip](https://pypi.python.org/pypi/pip):

```Shell
# make sure pip is upgraded
pip install --upgrade pip
# install jupyter notebook
pip install jupyter
# Run this inside ssd.pytorch
jupyter notebook
```

- Now navigate to `demo/demo.ipynb` at http://localhost:8888 (by default).

### Try the webcam demo
- Works on CPU (may have to tweak `cv2.waitkey` for optimal fps) or on an NVIDIA GPU
- This demo currently requires opencv2+ w/ python bindings and an onboard webcam
  * You can change the default webcam in `demo/live.py`
- Install the [imutils](https://github.com/jrosebr1/imutils) package to leverage multi-threading on CPU:
  * `pip install imutils`
- Running `python -m demo.live` opens the webcam and begins detecting!

### Try the mobile app demo
- Make sure you have installed Android Studio.
- Navigate to `../Android/`, download necessary Libraries and click `Run` button to use the application.
## References
- Jaco. "PyTorch Implementation of [1611.06440] Pruning Convolutional Neural Networks for Resource Efficient Inference." https://github.com/jacobgil/pytorch-pruningReferences
- Implementation of Variants of SSD Model. https://github.com/lzx1413/PytorchSSD
- Useful links of explanation for Mobilenet v2. http://machinethink.net/blog/mobilenet-v2/
- Useful links of explanation for Mobilenet v1. http://machinethink.net/blog/googles-mobile-net-architecture-on-iphone/
- Wei Liu, et al. "SSD: Single Shot MultiBox Detector." [ECCV2016]((http://arxiv.org/abs/1512.02325)).
- [Original Implementation (CAFFE)](https://github.com/weiliu89/caffe/tree/ssd)

