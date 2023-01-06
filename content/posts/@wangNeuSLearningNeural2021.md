---
title: "NeuS: Learning Neural Implicit Surfaces by Volume Rendering for
  Multi-view Reconstruction"
alias: wangNeuSLearningNeural2021
tags:
  - NeRF
  - SDF
rating: ⭐⭐⭐
shared: true
ptype: article
draft: false
date: 2022-12-27
---


# NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction
<cite>* Authors: [[Peng Wang]], [[Lingjie Liu]], [[Yuan Liu]], [[Christian Theobalt]], [[Taku Komura]], [[Wenping Wang]]</cite>


* [Local library](zotero://select/items/1_PBN526J8)

***

### 初读印象

comment:: NeuS 使用符号距离函数 SDF 对三维物体表面进行表示，提出一个 sigmoid 导数形式的 s-density 体渲染方法使得训练结果是 sdf 形式， 同时提出无偏的权重函数处理射线经过多表面的问题。


### Note

s-density 原理是 $\phi$ 是: 我们想从 SDF 获得density， 而 density 在物体的表面应该达到最大，而物体表面对应的 SDF 应该为 0， 所以应该选择一个在 0 处达到峰值的单峰函数, 比如文中选择的 $\phi_s$
![](https://markdown-imagebed.oss-cn-beijing.aliyuncs.com/imgs/202210121506811.png)

权重函数需要满足两个条件
1. 无偏性
	在物体表面时候权重应该最大， 这样能射线经过 SDF 的 0 面（也就是预测的表面）对像素的颜色贡献最多
3. 遮挡感知能力
   当一个射线上有两个深度值 $t_0$ 和 $t_1$ 对应相同的 sdf 值 ， 那么离 view point 更近的那个点应该贡献更多最终颜色， 原因是这种情况预示着这条射线经过了两个表面， 那个离观察位置更近的表面颜色更加重要。

---

### TL;DR

- for reconstructing objects and scenes with high ﬁdelity from 2D image inputs.
- Existing neural surface reconstruction approaches require foreground mask as supervision, easily get trapped in local minima, and therefore struggle with the reconstruction of objects with severe self-occlusion or thin structures
- NeRF extracting high-quality surfaces isdifficult because not sufﬁcient surface constraints in the representation
- NeuS represent a surface as the zero-level set of a signed distance function (SDF) and develop a new volume rendering method to train a neural SDF representation
- conventional volume rendering method causes inherent geometric errors (i.e. bias) for surface reconstruction
- propose a new formulation that is free of bias in the ﬁrst order of approximation, thus leading to more accurate surface reconstruction even without the mask supervision

### Inroduction
IDR 
- The surface rendering method used in IDR only considers a single surface intersection point for each ray produces impressive reconstruction results, but it fails to reconstruct objects with complex structures that causes abrupt depth changes.

NeRF
- surface extracted as a level-set surface of the density ﬁeld learned by NeRF contains conspicuous noise in some planar regions

NeuS 
- uses the signed distance function (SDF) for surface representation
- uses a novel volume rendering scheme to learn a neural SDF representation
- NeuS is capable of reconstructing complex 3D objects and scenes with severe occlusions and delicate structures, even __without foreground masks as supervision__

### Method
#### Rendering Procedure
##### Scene representation
1. $f: \mathbb{R}^3 \rightarrow \mathbb{R}$ that maps a spatial position $x \in \mathbb{R}^3$ to its signed distance to the object
2. $c: \mathbb{R}^3 \times \mathbb{S}^2 \rightarrow \mathbb{R}^3$ that encodes the color associated with a point $x \in R^3$ and a viewing direction  $v \in \mathbb{S}^2$

The surface $\mathcal{S}$ of the object is represented by the zero-level set of its SDF

$$
\mathcal{S}=\left\{\mathbf{x} \in \mathbb{R}^3 \mid f(\mathbf{x})=0\right\}
$$

- In order to apply a volume rendering method to training the SDF network we first introduce S-density: a probability density function $\phi_s{(f(x))}$ where $\phi_s(x)=s e^{-s x} /\left(1+e^{-s x}\right)^2$ is derivative of the Sigmoid function $\Phi_s(x)=\left(1+e^{-s x}\right)^{-1}$ , $\phi_s(x)=\Phi_s^{\prime}(x)$
- the standard deviation of $\phi_s(x)$ is given by 1/s , which is also a trainable parameter, that is, 1/s approaches to zero as the network training converges.

- the zero-level set of the network-encoded SDF is expected to represent an accurately reconstructed surface S, with its induced S-density φ s (f(x)) assuming prominently high values near the surface.

##### Rendering

given  a pixel, the ray emmited from the pixel as  $\{\mathbf{p}(t)=\mathbf{o}+t \mathbf{v} \mid t \geq 0\}$ . We accumulate the colors along the ray by

$$
C(\mathbf{o}, \mathbf{v})=\int_0^{+\infty} w(t) c(\mathbf{p}(t), \mathbf{v}) \mathrm{d} t
$$

where $C(o, v)$ is the output color for this pixel, $w(t)$ a weight for the point $p(t)$ , and $c(p(t), v)$ the color at the point $p$ along the viewing direction $v$ . 

__Requirements on weight function__
1. Unbiased
   -  $w(t)$ attains a locally maximal value at a surface intersection point $p(t^*)$ ,  the point $p(t^*)$ is one the zero-level of the $SDF(x)$
   -  guarantees $w(t)$ that the intersection of the camera ray with the zero-level set of SDF contributes most to the pixel color
2. Occlusion-aware
   - Given any two depth values $t_0$ and $t_1$ satisfying $f(t_0) = f(t_1)$, $w(t_0) > 0$ , $w(t_1 ) > 0$, and $t_0 < t_1$ , there is $w(t_0 ) > w(t_1 )$. That is, when two points have the same SDF value (thus the same SDF-induced S-density value), the point nearer to the view point should have a larger contribution to the ﬁnal output color than does the other point.
   - when a ray sequentially passes multiple surfaces, the rendering procedure will correctly use the color of the surface nearest to the camera to compute the output color.
 ![](https://markdown-imagebed.oss-cn-beijing.aliyuncs.com/imgs/202210121544636.png)
 $w(t)$
 
1. Naive solution of $w(t)$  : $w(t)=T(t) \sigma(t)$ where $T(t)=\exp \left(-\int_0^t \sigma(u) \mathrm{d} u\right)$
   occlusion-aware but is biased
2. our solution
   $$
   w(t)=T(t) \rho(t), \quad \text { where } T(t)=\exp \left(-\int_0^t \rho(u) \mathrm{d} u\right)
  $$
  $$
\rho(t)=\max \left(\frac{-\frac{\mathrm{d} \Phi_s}{\mathrm{~d} t}(f(\mathbf{p}(t)))}{\Phi_s(f(\mathbf{p}(t)))}, 0\right)
$$


 ![](https://markdown-imagebed.oss-cn-beijing.aliyuncs.com/imgs/202210121609200.png)
 

#### Training

$n$: point sampling size
$m$: batch size


$$
\mathcal{L}=\mathcal{L}_{\text {color }}+\lambda \mathcal{L}_{\text {reg }}+\beta \mathcal{L}_{\text {mask }}
$$

**color loss**

$$
\mathcal{L}_{\text {color }}=\frac{1}{m} \sum_k \mathcal{R}\left(\hat{C}_k, C_k\right)
$$

$\mathcal{R}$ is L1 loss

**Eikonal term**

$$
\mathcal{L}_{r e g}=\frac{1}{n m} \sum_{k, i}\left(\left\|\nabla f\left(\hat{\mathbf{p}}_{k, i}\right)\right\|_2-1\right)^2
$$

**mask loss**

$$
\mathcal{L}_{\text {mask }}=\operatorname{BCE}\left(M_k, \hat{O}_k\right)
$$

##### Hierarchical sampling


