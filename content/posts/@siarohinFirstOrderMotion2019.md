---
title: First Order Motion Model for Image Animation
alias: siarohinFirstOrderMotion2019
tags:
  - face-driving
rating: ⭐⭐⭐
share: true
ptype: article
date: 2022-12-27
---


# First Order Motion Model for Image Animation
<cite>* Authors: [[Aliaksandr Siarohin]], [[Stéphane Lathuilière]], [[Sergey Tulyakov]], [[Elisa Ricci]], [[Nicu Sebe]]</cite>


* [Local library](zotero://select/items/1_TFWKNBWU)

***

### 初读印象

comment:: FOMM 是 Monkey-Net 的续作， 解决 Monkey-Net 对复杂动作处理能力比较弱的问题。具体方式是在估计光流时除了关键点本身之外还引入了局部仿射变换以获取精细的形变，同时会预估遮挡区域，指导inpainting模块进行填补。

## Method
![](https://markdown-imagebed.oss-cn-beijing.aliyuncs.com/imgs20220831235007.png)

整体架构如上图所示， 它主要由 motion estimation 模块和 image generation 模块构成。其中 motion estimation 模块用于预测驱动视频帧 $\mathcal{D}$ 到源图像 $\mathcal{S}$ 的 dense motion  field。 dense motion field 也叫反向光流，用 $\mathcal{T}_{\mathbf{S} \leftarrow \mathbf{D}}: \mathbb{R}^2 \rightarrow \mathbb{R}^2$ 表示。

整体上分成三步
1. 预估 R 到 S ， D 到 R 的关键点以及对应的仿射变换参数 
2. 预估光流以及遮挡 mask
3. 生成驱动图像

### Local Afﬁne Transformations for Approximate Motion Description




