---
title: "CornerNet: Detecting Objects as Paired Keypoints"
alias: lawCornerNetDetectingObjects2020
tags:
  - detection
  - one-stage
  - anchor-free
rating: ⭐⭐
shared: true
ptype: article
draft: false
date: 2022-12-27
timezone: UTC+8
---


# CornerNet: Detecting Objects as Paired Keypoints
<cite>* Authors: [[Hei Law]], [[Jia Deng]]</cite>

* DOI: [10.1007/s11263-019-01204-1](https://doi.org/10.1007/s11263-019-01204-1)

* [Local library](zotero://select/items/1_INVERZU4)

***

### 初读印象

CornerNet 预测目标的左上和右下角点heatmap, 提取焦点的 embedding 进行角点对匹配来替代 anchor

## TL;DR
基于 anchor 的方法的缺点
1. 需要铺设大量的 anchors，但是只有少量是和 gt 相交的有用 anchors， 带来了严重的正负样本不均匀问题；
2. 引入大量的超参，`base_sizes` 、`ratios`、`scales`

为了解决上述问题提出 cornernet ，预测目标的左上和右下角点heatmap, 提取焦点的 embedding 进行角点对匹配来替代 anchor。

需要重点关注的有三个地方
1. detecting corners
2. grouping corners
3. corner pooling
4. hourglass network

## Method

![](https://markdown-imagebed.oss-cn-beijing.aliyuncs.com/imgs20210623233532.png)
整体框架如上

### detecting corners
在 backbone 后面接两个 CxHxW 的分支， C 为目标类别数（注意不包括背景类别），每个点的值代表该点位对应类别目标一个角点的可能性。对于每一个角点存在一个 gt 位置，其他位置都是 negative。**但是在训练的时候容忍 gt 周围 radius 范围的角点偏移误差，因为偏离了一点 gt 还是可能得到 iou 足够的预测框**。 定位准确度用这个式子表示： $
e^{-\frac{x^{2}+y^{2}}{2 \sigma^{2}}}
$ 其中 $\sigma=\frac{1}{3}radius$

基于此的角点 loss：

$$
L_{d e t}=\frac{-1}{N} \sum_{c=1}^{C} \sum_{i=1}^{H} \sum_{j=1}^{W}\left\{\begin{array}{cc}\left(1-p_{c i j}\right)^{\alpha} \log \left(p_{c i j}\right) & \text { if } y_{c i j}=1 \\ \left(1-y_{c i j}\right)^{\beta}\left(p_{c i j}\right)^{\alpha} \log \left(1-p_{c i j}\right) & \text { otherwise }\end{array}\right.
$$

$p_{cij}$ 为位置 (i,j) 处， 类别 c 的分数。

由于 CNN 最后输出是经过下采样的，最后的坐标映射回原图会有损失，为此网络增加对映射误差的预测：

$$
\boldsymbol{o}_{k}=\left(\frac{x_{k}}{n}-\left\lfloor\frac{x_{k}}{n}\right\rfloor, \frac{y_{k}}{n}-\left\lfloor\frac{y_{k}}{n}\right\rfloor\right)
$$

使用下面的 loss 来训练

$$
L_{o f f}=\frac{1}{N} \sum_{k=1}^{N} \operatorname{SmoothL} 1 \operatorname{Loss}\left(\boldsymbol{o}_{k}, \hat{\boldsymbol{o}}_{k}\right)
$$

### Grouping Corners
对于一张图有多个相同类别的目标的情形， 需要对预测的 corners 进行配对。
为每个预测的角点生成一个 embeding， 我们认为同一个目标的左上角和右下角 embeding 是近似的，所以通过比较 embeding 的距离可以确定哪些是配对的：

$$
\begin{array}{l}L_{\text {pull }}=\frac{1}{N} \sum_{k=1}^{N}\left[\left(e_{t_{k}}-e_{k}\right)^{2}+\left(e_{b_{k}}-e_{k}\right)^{2}\right] \\ L_{\text {push }}=\frac{1}{N(N-1)} \sum_{k=1}^{N} \sum_{j=1 \atop j \neq k}^{N} \max \left(0, \Delta-\left|e_{k}-e_{j}\right|\right)\end{array}
$$

$e_{tk}, e_{bk}, e_{k}$ 分别为左上角 embeding、右下角embedding， 左上右下embedding 的平均， $\Delta$ 设为 1。

### Corner Pooling
由于角点位置经常不包含目标，比如圆形目标检测框的角点周围都是背景信息， 为了确定上边界，需要从最左往最右看才能确定， 为了确定左边界，需要从上往下看才能确定。
作者提出了 Corner Pooling 的池化方法：

$$
\begin{array}{c}t_{i j}=\left\{\begin{array}{cc}\max \left(f_{t_{i j}}, t_{(i+1) j}\right) & \text { if } i<H \\ f_{t_{H j}} & \text { otherwise }\end{array}\right. \\ l_{i j}=\left\{\begin{array}{cl}\max \left(f_{l_{i j}}, l_{i(j+1)}\right) & \text { if } j<W \\ f_{l_{i W}} & \text { otherwise }\end{array}\right.\end{array}
$$

![](https://markdown-imagebed.oss-cn-beijing.aliyuncs.com/imgs20210624000037.png)

![](https://markdown-imagebed.oss-cn-beijing.aliyuncs.com/imgs20210624000140.png)


### Hourglass Network

![](https://markdown-imagebed.oss-cn-beijing.aliyuncs.com/imgs20210624000220.png)
总 loss：

$$
L=L_{d e t}+\alpha L_{p u l l}+\beta L_{p u s h}+\gamma L_{o f f}
$$

## Experiments

![](https://markdown-imagebed.oss-cn-beijing.aliyuncs.com/imgs20210624000431.png)

corner pooling 很重要，整体提升 2 个点，对大物体提升 3.6 个点

![](https://markdown-imagebed.oss-cn-beijing.aliyuncs.com/imgs20210624000528.png)

减小 gt 周围的惩罚对点数影响明显

![](https://markdown-imagebed.oss-cn-beijing.aliyuncs.com/imgs20210624000625.png)

Hourglass 很重要