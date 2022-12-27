---
title: "Libra R-CNN: Towards balanced learning for object detection"
alias: pangLibraRCNNBalanced2019
tags:
  - detection
  - two-stage
rating: ⭐⭐
share: true
ptype: article
date: 2022-12-27 23:31:01
---


# Libra R-CNN: Towards balanced learning for object detection
<cite>* Authors: [[Jiangmiao Pang]], [[Kai Chen]], [[Jianping Shi]], [[Huajun Feng]], [[Wanli Ouyang]], [[Dahua Lin]]</cite>

* DOI: [10.1109/CVPR.2019.00091](https://doi.org/10.1109/CVPR.2019.00091)

* [Local library](zotero://select/items/1_XPF26ZHI)

***

### 初读印象

comment:: 从采样、feature map、损失函数三个角度解决目标检测任务样本不均衡问题， 提出了对应的 IoU-sampling、balanced feature pyramid 和 balanced L1-loss 三个组件。

## TL;DR
从采样、feature map、损失函数三个角度解决目标检测任务样本不均衡问题， 提出了对应的 IoU-sampling、balanced feature pyramid 和 balanced L1-loss 三个组件。

## Method

一个好的检测器依赖于下面三点：
(1) whether the selected region samples are representative, (2) whether the extracted visual features are fully utilized, and (3) whether the designed objective function is optimal.

然而，很少有工作能同时做好这三点， 比如：
OHEM 能将模型注意力转至难样本，但是同时会过多关注 noise labels、 带来更多的计算负担； Focal Loss对一阶段模型很有用，但是对 RCNN 方法作用有限，因为两阶段模型已经把大部分简单样本过滤掉了。

作者提出的方法整体框架：

![](https://markdown-imagebed.oss-cn-beijing.aliyuncs.com/imgs20210704172809.png)


### IoU balanced sampling

作者通过实验发现超过 60% 的难负样本和 gt 的 iou 大于 60%, 但是随即采样只获得了 30% 这个区间的样本。

假设需要的负样本数量是 N，候选样本数量是 M， IoU balaced samplig 把 N 个样本按和gt 的iou均匀切分成 K 个 bin， 然后在每个bin里均匀采样，公式如下：

$$
p_{k}=\frac{N}{K} * \frac{1}{M_{k}}, k \in[0, K)
$$

### Balanced Feature Pyramid

![](https://markdown-imagebed.oss-cn-beijing.aliyuncs.com/imgs20210704173326.png)

如上图所示，先将 FPN 所有feature integrate 起来综合各个 level 的语义信息，

$$
C=\frac{1}{L} \sum_{l=l_{m i n}}^{l_{m a x}} C_{l}
$$

然后用 non-local module 做refine，最后基于这个refine结果上/下采样回到原来的 levels 进行后续步骤


### Balanced L1 Loss
目标检测任务需要同时处理分类和回归两个问题， 然而如果使用 L1 loss 的话，难样本由于与 gt 距离比较远，产生的 loss 比较大， 所以作者提出类似于 smooth l1 loss 的裁减方法。

首先定义 loss < 1.0 的为 inliers, 大于等于 1.0 的为 outlier,

![](https://markdown-imagebed.oss-cn-beijing.aliyuncs.com/imgs20210704174031.png)

如图5(a) 所示， smooth l1 loss 对于loss 大于 1.0 的样本，其梯度直接变成常数1.0, smoothed l1 loss 同样如此， 它的梯度定义如下：

$$
\frac{\partial L_{b}}{\partial x}=\left\{\begin{array}{ll}\alpha \ln (b|x|+1) & \text { if }|x|<1 \\ \gamma & \text { otherwise }\end{array}\right.
$$

$\alpha$ 越小，inlier 样本梯度越大，而 $\gamma$ 调整针对 outlier 的梯度上限。最后的形式如下：

$$
L_{b}(x)=\left\{\begin{array}{ll}\frac{\alpha}{b}(b|x|+1) \ln (b|x|+1)-\alpha|x| & \text { if }|x|<1 \\ \gamma|x|+C & \text { otherwise }\end{array}\right.
$$

$$
\alpha \ln (b+1)=\gamma
$$
 



