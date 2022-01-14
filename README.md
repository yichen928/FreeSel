# Towards General and Efficient Active Learning
This repo is official code implementation for the paper "Towards General and Efficient Active Learning". You can find the paper on [arXiv](https://arxiv.org/abs/2112.07963). Code is coming soon.

Existing active learning work follows a cumbersome pipeline by repeating the time-consuming model training and batch data selection multiple times on each dataset separately. We challenge this status-quo by proposing a novel general and efficient active learning (GEAL) method in this paper. Utilizing a publicly available model pre-trained on a large dataset, our method can conduct data selection processes on different datasets with a single-pass inference of the same model.

<div align="center">
    <img src="figs/pipeline.jpg", width="600">
</div>

Our method is significantly more efficient than prior arts by hundreds of times, while the performance is competitive or even better than methods following the traditional pipeline.
<div align="center">
    <img src="figs/efficiency.jpg", width="600">
</div>

<div align="center">
    <img src="figs/performance.jpg", width="600">
</div>


## Citation
If you find our research helpful, please consider cite it as:
```
@article{xie2021towards,
  title={Towards General and Efficient Active Learning},
  author={Xie, Yichen and Tomizuka, Masayoshi and Zhan, Wei},
  journal={arXiv preprint arXiv:2112.07963},
  year={2021}
}
```
