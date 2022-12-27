---
title: "Ray Priors through Reprojection: Improving Neural Radiance Fields for
  Novel View Extrapolation"
alias: zhangRayPriorsReprojection2022
tags:
  - NeRF
  - CVPR2022
rating: ⭐⭐
share: true
ptype: article
date: 2022-12-27
---


# Ray Priors through Reprojection: Improving Neural Radiance Fields for Novel View Extrapolation
<cite>* Authors: [[Jian Zhang]], [[Yuanqing Zhang]], [[Huan Fu]], [[Xiaowei Zhou]], [[Bowen Cai]], [[Jinchi Huang]], [[Rongfei Jia]], [[Binqiang Zhao]], [[Xing Tang]]</cite>

* DOI: [10.1109/CVPR52688.2022.01783](https://doi.org/10.1109/CVPR52688.2022.01783)

* [Local library](zotero://select/items/1_LESBZC5J)

***

### 初读印象

comment:: 解决测试视角和训练视角相差很大时 NeRF 效果不好的问题


![](https://markdown-imagebed.oss-cn-beijing.aliyuncs.com/imgs/202210311518653.png)

![](https://markdown-imagebed.oss-cn-beijing.aliyuncs.com/imgs/202210311634040.png)

### 文章骨架
%%创新点到底是什么?%%
novelty:: 使用随机光线投射(random ray casting (RRC)) 以真实光线为中心在半球上构造伪标注的新射线，从而扩展可用视角；另一个改进是提出 Ray Atlas (RA) 以全局的射线方向替代单一的射线方向， 使得整个 NeRF 框架对视角扩展更加友好。

%%有什么意义？%%
significance:: 解决测试视角和训练视角相差很大时 NeRF 效果不好的问题

%%有什么潜力?%% 
potential:: 稀疏视角下的渲染


## TL;DR

本文是阿里淘系 CVPR 2022 的 paper， 主要解决的是 NeRF 在对拍摄物体视角覆盖范围不够严密时效果不好的问题。 具体方法是使用随机光线投射(random ray casting (RRC)) 以真实光线为中心在半球上构造伪标注的新射线，从而扩展可用视角；另一个改进是提出 Ray Atlas (RA) 以全局的射线方向替代单一的射线方向， 使得整个 NeRF 框架对视角扩展更加友好。

## 出发点

![](https://markdown-imagebed.oss-cn-beijing.aliyuncs.com/imgs/202210311518653.png)
原始的 NeRF 要求拍摄对象尽可能覆盖全面， 如果满足不了的话视角稀疏区域的重建对象往往不如人意。以 Figure 2 为例， 属于训练集的射线 $r_1$ 和属于测试集的 $r_2$ 虽然都穿过 $v$ ，但是他们的视角点距离比较远， 导致射线穿过的区域很不一样，$r_1$ 穿过的大部分区域属于空旷区域，而 $r_2$ 穿过了很多不透明的区域， 由于 NeRF 光线行进积分的特性两者渲染出来的结果也会很不一样， 图中 $r_2$ 不再训练集的视角范围内， 所以得到的渲染结果不是很好。

## Random Ray Casting (RRC)

如 Figure 2 右图所示， 以图片中射线到物体交点 v 为球心作一个半球面， 然后以 ov 为中心射线在 $[-\eta,\eta]$ 范围内随机改变方位角 $\Delta\varphi$  和高程角 $\Delta\theta$ 生成同样经过 $v$ 的以 $o^\prime$ 为中心的新射线 $r^\prime$ ， 对于这个新生成的射线，以 $r$ 的颜色 $I(r)$ 作为伪标签 ${\hat{I}}(\mathbf{r}^{\prime})$ 。这样， 对于 $v$ 我们不仅有拍摄图对应的射线， 还有我们生成其他视角对应的伪标签射线， 可以大大丰富模型对视角的鲁棒性。

## Ray Atlas (RA)

我们知道 NeRF 的输入是包含源点 $o$ 和射线方向 $v$ 的五元组， 但是因为训练和测试的射线分布是有明显区别的， 这个带射线方向的输入对视角扩展并不友好。针对这个问题作者提出 RA 这种全局射线方向替代单一射线方向。

首先， 对于每张图片 $I$ 获取每一个像素位置对应的射线方向作为射线集 $R(I)$ , 然后用预训练 NeRF 获取一个粗糙的 3D mesh (R3DM) ， 对于每一个顶点 ${\cal V}\,=\,(x,y,z)$ 定义它的 全局射线方向如下

$$ {\bar{\mathbf{d}}}_{V}=\frac{1}{L}\sum_{i=1}^{L}R(I_{i})[\nu_{u v}(I_{i})], $$
$$ \mathcal{V}_{w v}(I_{i})=\frac{1}{z}K{\mathcal{T}_{w2c}}(I_{i})V $$

其中 $K$ 是相机内参, $T_{w2c}$ 是世界坐标和相机坐标的转换矩阵，$L$ 是包括该顶点的所有图像。图例如下：
![](https://markdown-imagebed.oss-cn-beijing.aliyuncs.com/imgs/202210311634040.png)

在训练阶段在原始射线方向而不是在 $\bar{d}$ 方向上采样， 因为作者发现那样做会造成训练不稳定。测试阶段五元组中的射线方向 $d$ 用 $\hat{d}$ 替代：

$$ {\mathfrak{c}}=F_{c}(\bar{d},F_{\sigma}({\bf x})) $$


## 训练

训练分两阶段
1. 训练原始的 NeRF $T_1$ 个 iterations。
2. 引入 RRC 和 RA 继续 finetune $T_2$ 个 iterations。

训练时并不是所有射线会使用到 RRC 和 RA， 而是以 0.7 和 0.5 的概率应用， 同时 RRC 射线扩充角度的范围 $\eta$ 设为 30°。


## 实验结果

![](https://markdown-imagebed.oss-cn-beijing.aliyuncs.com/imgs/202210311641057.png)

![](https://markdown-imagebed.oss-cn-beijing.aliyuncs.com/imgs/202210311641211.png)

RRC 可以带来 3.5 个点的 PSNR 提升， RA 可以带来 1.24 个点的 PSNR 提升。



