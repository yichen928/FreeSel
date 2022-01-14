# Towards General and Efficient Active Learning
This repo is official code implementation for the paper "Towards General and Efficient Active Learning". You can find the paper on [arXiv](https://arxiv.org/abs/2112.07963). Code is coming soon.

Existing active learning work follows a cumbersome pipeline by repeating the time-consuming model training and batch data selection multiple times on each dataset separately. We challenge this status-quo by proposing a novel general and efficient active learning (GEAL) method in this paper. Utilizing a publicly available model pre-trained on a large dataset, our method can conduct data selection processes on different datasets with a single-pass inference of the same model.
![image](https://user-images.githubusercontent.com/48796750/149582566-94fdae9e-a2fc-48de-9754-0f7ee9e03a62.png)

Our method is significantly more efficient than prior arts by hundreds of times, while the performance is competitive or even better than methods following the traditional pipeline.
![image](https://user-images.githubusercontent.com/48796750/149582753-bfc9ab53-334d-4d08-ace3-60f14c1184b6.png)
![image](https://user-images.githubusercontent.com/48796750/149582819-e5b5bb60-7b35-43bd-8ec6-829e5e9a56a3.png)


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
