---
title: Bridging the Gap Between Anchor-Based and Anchor-Free Detection via
  Adaptive Training Sample Selection
alias: zhang2020
tags:
  - detection
  - label_assign
  - one-stage
rating: ⭐⭐⭐
shared: true
ptype: article
draft: false
date: 2022-12-27
timezone: UTC+8
---


# Bridging the Gap Between Anchor-Based and Anchor-Free Detection via Adaptive Training Sample Selection
<cite>* Authors: [[Shifeng Zhang]], [[Cheng Chi]], [[Yongqiang Yao]], [[Zhen Lei]], [[Stan Z. Li]]</cite>

* DOI: [10.1109/CVPR42600.2020.00978](https://doi.org/10.1109/CVPR42600.2020.00978)

* [Local library](zotero://select/items/1_QZQNGHFX)

***

### 初读印象

ATSS 提出一种基于训练时匹配框的 std 和 mean 动态确定阈值的 label assign 方法

### 文章骨架
%%创新点到底是什么?%%
novelty:: 分析了当前一些 label assign 的区别，提出了根据对象的统计特征自动选择正负样本的方法

%%有什么意义？%%
significance:: 提升了模型检测精度

%%有什么潜力?%% 
potential:: 





## 贡献
- 指出anchor-free和anchor-based方法的根本差异主要来源于正负样本的选择
- 提出ATSS( Adaptive Training Sample Selection)方法来根据对象的统计特征自动选择正负样本
- 证明每个位置设定多个anchor是无用的操作
- 不引入其它额外的开销，在MS COCO上达到SOTA

## 分析
将 RetinaNet Anchor 数量设为 1 （1:1 正放形 anchor） 加入 GroupNorm、GIoU Loss、In Gt Box、Centerness 和 Scalar 将其和 FCOS 对齐，  RetinaNet 相比 FCOS 还要低 0.8 个点。 此时主要差异由两个：1. 正负样本定义； 2. 预测值到 bbox 的转换方式

1. 正负样本定义
   ![](https://markdown-imagebed.oss-cn-beijing.aliyuncs.com/imgs20210615232952.png)
   RetinaNet 设定两个
2. 预测方式

![](https://markdown-imagebed.oss-cn-beijing.aliyuncs.com/imgs/20210623150436.png) 


