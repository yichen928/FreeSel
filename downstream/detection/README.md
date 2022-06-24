## Downstream tasks: object detection

The object detection code is borrowed from **[mmdetection](https://github.com/open-mmlab/mmdetection)**. Please install this repo according to their instructions.

### Usage

1. Put our config file `ssd300_voc0712_al.py` into `mmdetection/configs/pascal_voc`, which includes the same training hyperparameters with prior active learning work.
2. Customize the training split files with the ones generated from  our `data_selection` code by specifying `--cfg-options data.train.dataset.ann_file=`.

##### Command for reference:

```
python tools/train.py configs/pascal_voc/ssd300_voc0712_al.py --work-dir work_dirs/ssd300_voc0712_al_dino_vits16_GEAL_3000 --cfg-options data.train.dataset.ann_file=data/VOCdevkit/VOC2007/ImageSets/Main/trainval_dino_vits16_GEAL_3000.txt,data/VOCdevkit/VOC2012/ImageSets/Main/trainval_dino_vits16_GEAL_3000.txt
```

