---
title: Training data-efficient image transformers & distillation through attention
alias: touvron2021
tags:
  - ViT
  - transformer
  - classification
rating: ⭐⭐
share: true
ptype: article
date: 2022-12-27
---


# Training data-efficient image transformers & distillation through attention
<cite>* Authors: [[Hugo Touvron]], [[Matthieu Cord]], [[Matthijs Douze]], [[Francisco Massa]], [[Alexandre Sablayrolles]], [[Hervé Jégou]]</cite>


* [Local library](zotero://select/items/1_GKGVSQQ7)

***

### 初读印象

comment:: DeiT 提出了一种适合 ViT 的蒸馏方法， 即除了 [CLS] token 之外增加了一个类似的用于蒸馏的 distillation token， 他们的区别是 CLS token 的监督信号的 groundtruth， 而 distillation token

![](https://markdown-imagebed.oss-cn-beijing.aliyuncs.com/imgs/202212271411628.png)

### 文章骨架
%%创新点到底是什么?%%
novelty:: 即除了 [CLS] token 之外增加了一个类似的用于蒸馏的 distillation token， 他们的区别是 CLS token 的监督信号的 groundtruth， 而 distillation token
的监督信号是 teacher 的 logits 或预测标签。

%%有什么意义？%%
significance:: 提出了一种适合 ViT 的蒸馏方法

%%有什么潜力?%% 
potential:: 减小了 ViT 的训练难度， 提高了 ViT 小模型的性能



![title](https://markdown-imagebed.oss-cn-beijing.aliyuncs.com/imgs/20210508153307.png)


文章的实验有几个有趣的发现：
1. 使用卷积网络作为 teacher 效果好于其他 vit 模型作为 teacher
2. 使用 hard-label 效果要优于 soft-label

   - soft-label
      $$ \mathcal{L}_{\text {global }}=(1-\lambda) \mathcal{L}_{\mathrm{CE}}\left(\psi\left(Z_{\mathrm{s}}\right), y\right)+\lambda \tau^{2} \mathrm{KL}\left(\psi\left(Z_{\mathrm{s}} / \tau\right), \psi\left(Z_{\mathrm{t}} / \tau\right)\right) $$
      其中 $\psi$ 为 softmax， $Z_s$ 和 $Z_t$ 为 student 和 teacher 的 logits， $\tau$ 为温度系数
   - hard-label
    $$ \mathcal{L}_{\text {global }}^{\text {hardDistill }}=\frac{1}{2} \mathcal{L}_{\mathrm{CE}}\left(\psi\left(Z_{s}\right), y\right)+\frac{1}{2} \mathcal{L}_{\mathrm{CE}}\left(\psi\left(Z_{s}\right), y_{\mathrm{t}}\right) $$
    其中 $y_t$ 为 teacher 预测的标签 
