# Batch Normalization with Enhanced Linear Transformation
Link to paper: https://arxiv.org/pdf/2011.14150v1.pdf
*note: I am not an author. This is just my implementation of the paper combined with some experiments.*
*note: You will also find experimental resnet models and resnet variants. Such as xsresnet, resnest.*

## Abstract 
*Batch normalization (BN) is a fundamental unit in modern deep networks, in which a linear transformation module
was designed for improving BN’s flexibility of fitting complex data distributions. In this paper, we demonstrate properly enhancing this linear transformation module can effectively improve the ability of BN. Specifically, rather than
using a single neuron, we propose to additionally consider
each neuron’s neighborhood for calculating the outputs of
the linear transformation. Our method, named BNET, can
be implemented with 2–3 lines of code in most deep learning libraries. Despite the simplicity, BNET brings consistent performance gains over a wide range of backbones
and visual benchmarks. Moreover, we verify that BNET
accelerates the convergence of network training and enhances spatial information by assigning the important neurons with larger weights accordingly. The code is available
at https://github.com/yuhuixu1993/BNET.*

**Link to their github it posted above**.

## Dataset
I am using the FastAI Imagenette dataset which can be found and downloaded here: https://github.com/fastai/imagenette

## Conclusion
Run experiments notebook to replicate my results. From what I have noticed, training does seem to be a lot more stable when compared to using just BatchNorm2d. 

I also saw greater convergence.

**Still need to update this readme with more results and tables.** 

## Citation
```
@article{BNET,
  title   = {Batch Normalization with Enhanced Linear Normalization},
  author  = {Xu, Yuhui and Xie, Lingxi and Xie, Cihang and Mei, Jieru and
             Qiao, Siyuan and Wei, Shen and Xiong, Hongkai and Alan, Yuille},
  journal= {arXiv preprint arXiv:2011.14150},
  year={2020}
}
```