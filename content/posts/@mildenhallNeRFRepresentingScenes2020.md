---
title: "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
alias: mildenhallNeRFRepresentingScenes2020
tags:
  - NeRF
rating: ⭐⭐⭐
share: true
ptype: article
date: 2022-12-27
---


# NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis
<cite>* Authors: [[Ben Mildenhall]], [[Pratul P. Srinivasan]], [[Matthew Tancik]], [[Jonathan T. Barron]], [[Ravi Ramamoorthi]], [[Ren Ng]]</cite>


* [Local library](zotero://select/items/1_4WIUKH3N)

***

### 初读印象

comment:: NeRF 提出了一种使用 5D 输入(射线起点3维坐标以及视线2维方向) 到三维空间密度和颜色的三维场景的神经辐射场表示


[都2022年了，我不允许你还不懂NeRF - mathfinder的文章 - 知乎](https://zhuanlan.zhihu.com/p/569843149)

### Introduction
What is Nerf:
We represent a static scene as a continuous 5D function that outputs the radiance emitted in each direction (θ, φ ) at each point (x, y, z) in space, and a density at each point which acts like a diﬀerential opacity controlling how much radiance is accumulated by a ray passing through (x, y, z).

optimize: minimizing the error between each observed image and the corresponding views rendered from our representation


shortcomming of basic nerf representation:
1. not converge to a suﬃciently high-resolution representation
2. ineﬃcient in the required number of samples per camera ray.

addressed by:
transforming input 5D coordinates with a positional encoding that enables the MLP to represent higher frequency functions, and we propose a hierarchical sampling procedure to reduce the number of queries required to adequately sample this high-frequency scene representation.

#### Contributions
1. representing continuous scenes with complex geometry and materials as 5D neural radiance ﬁelds, parameterized as basic MLP networks.
2. A diﬀerentiable rendering procedure based on classical volume rendering techniques,
3. A positional encoding to map each input 5D coordinate into a higher dimensional space,

### Related Work

- Neural 3D shape representations
  shortcoming: 
  1. requirement of access to ground truth 3D geometry,
  2. limited to simple shapes with low geometric complexity,
- View synthesis and image-based rendering
  - gradient-based mesh optimization
	  1. optimization based on image is often difficult
	  2. requires a template mesh with ﬁxed topology
  - volumetric representations
	  1. scale to higher resolution imagery is fundamentally limited by poor time and space complexity
 
### Neural Radiance Field Scene Representation

![](https://markdown-imagebed.oss-cn-beijing.aliyuncs.com/imgs/202209191121835.png)

inputs:
- 3D location $x = (x, y, z)$
- 2D viewing direction $(\theta, \phi )$, expressed by 3D Cartesian unit vector $d$
outputs:
- color $c = (r, g, b)$
- volume density $σ$

MLP network $F_\Theta: (\mathbf{x}, \mathbf{d}) \rightarrow(\mathbf{c}, \sigma)$

the network predict
1. density $\sigma$ as a function of only the location $x$
2. the RGB color $c$ to be predicted as a function of both location and viewing direction

The MLP $F_\Theta$:
1. processes the input 3D coordinate $x$ with 8 fully-connected layers (using ReLU activations and 256 channels per layer), and outputs $σ$ and a 256-dimensional feature vector.
2. This feature vector is then concatenated with the camera ray’s viewing direction and passed to one additional fully-connected layer (using a ReLU activation and 128 channels) that output the view-dependent RGB color


