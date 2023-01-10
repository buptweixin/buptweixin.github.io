---
title: What do Vision Transformers Learn? A Visual Exploration
alias: ghiasiWhatVisionTransformers2022
tags:
  - ViT
  - transformer
rating: ⭐⭐
shared: true
ptype: article
draft: false
date: 2022-12-28
timezone: UTC+8
---


# What do Vision Transformers Learn? A Visual Exploration
<cite>* Authors: [[Amin Ghiasi]], [[Hamid Kazemi]], [[Eitan Borgnia]], [[Steven Reich]], [[Manli Shu]], [[Micah Goldblum]], [[Andrew Gordon Wilson]], [[Tom Goldstein]]</cite>


* [Local library](zotero://select/items/1_RRRTNCJP)

***

### 初读印象

提出了一种可视化方法解决 ViT 可视化困难的问题，基于此有几个发现：1. CLIP 这类带语言模型监督的 ViT 是由语义信息而不是直接的视觉特征激活；2. 相比 CNNs ViTs 的预测更少依赖于高频信息；3. 和 CNNs 类似随着层数加深特征由抽象到具象；4. 除了最后一层以外都包含空间信息，最后一层更像是抛弃空间信息的全局卷积。

![image.png](https://markdown-imagebed.oss-cn-beijing.aliyuncs.com/imgs/202212281606191.png)


