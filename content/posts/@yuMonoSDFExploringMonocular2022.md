---
title: "MonoSDF: Exploring Monocular Geometric Cues for Neural Implicit Surface
  Reconstruction"
alias: yuMonoSDFExploringMonocular2022
tags:
  - SDF
rating: ⭐
shared: true
ptype: article
draft: false
date: 2022-12-27
timezone: UTC+8
---


# MonoSDF: Exploring Monocular Geometric Cues for Neural Implicit Surface Reconstruction
<cite>* Authors: [[Zehao Yu]], [[Songyou Peng]], [[Michael Niemeyer]], [[Torsten Sattler]], [[Andreas Geiger]]</cite>


* [Local library](zotero://select/items/1_TZEV6ESM)

***

### 初读印象

NeuS [[@wangNeuSLearningNeural2021]]  基础上增加了单目深度和 normal 约束， 提升稀疏图片和低纹理条件下的重建效果。

![](https://markdown-imagebed.oss-cn-beijing.aliyuncs.com/imgs/202212271128734.png)

### 文章骨架
%%创新点到底是什么?%%
novelty:: 增加单目深度和 normal 约束

%%有什么意义？%%
significance:: 增强 NeuS 在低纹理和稀疏视角下的重建效果

%%有什么潜力?%% 
potential::  1. 受单目深度和法向图准确性影响比较大；2. 可以增加其他的约束，比如平面、边缘、遮挡等；3. 当前受限于 omnidata model 的 384x384 的输入， 可以开发更大的输入分辨率模型；4. 联合优化场景表示和相机参数

同时优化场景表示和相机参数 [[@azinovicNeuralRGBDSurface2022]] [[@zhu2022]]




