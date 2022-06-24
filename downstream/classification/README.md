## Downstream tasks: image classification

The image classification code is borrowed from **[mmclassification](https://github.com/open-mmlab/mmclassification)**. Please install this repo according to their instructions.

### Usage

1. Put our config file `resnet18_bs128_cifar10.py` into `mmclassification/configs/resnet`, which includes the same training hyperparameters with prior active learning work.
2. Customize the training split files with the ones generated from  our `data_selection` code by specifying `--cfg-options data.train.ann_file=`.

##### Command for reference:

```
python tools/train.py configs/resnet/resnet18_bs128_cifar10.py --work-dir work_dirs/ssd300_voc0712_al_GEAL_3000 --cfg-options data.train.dataset.ann_file=data/cifar10/splits/CIFAR10_dino_vits16_GEAL_3000.json
```

