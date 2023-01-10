---
title: "Free-HeadGAN: Neural Talking Head Synthesis with Explicit Gaze Control"
alias: doukasFreeHeadGANNeuralTalking2022
tags:
  - face-driving
rating: ⭐⭐
shared: true
ptype: article
draft: false
date: 2022-12-27
timezone: UTC+8
---


# Free-HeadGAN: Neural Talking Head Synthesis with Explicit Gaze Control
<cite>* Authors: [[Michail Christos Doukas]], [[Evangelos Ververas]], [[Viktoriia Sharmanska]], [[Stefanos Zafeiriou]]</cite>


* [Local library](zotero://select/items/1_TU5KFM6U)

***

### 初读印象

Free-HeadGAN

### TL;DR

- canonical 3D key-point estimator + gaze estimation network + generator
- use a neural network that learns to predict dense optical ﬂow
- does not predict canonical key-points directly

### contributes
- We release HeadGAN from its 3DMM priors, using sparse 3D landmarks
- a module that performs disentanglement of identity, expression and pose through the computation of canonical 3D key-points.
- propose a network that regresses 3D meshes of the eyes. We use these meshes to obtain the direction of gaze, which is then used to condition image synthesis

### methodology
- Each component of FreeHeadGAN is trained separately on its individual task

#### canonical keypoints

![](https://markdown-imagebed.oss-cn-beijing.aliyuncs.com/imgs/202209142131701.png)

- Given a portrait image, our goal here is to extract a set of key-points in a canonical space, which are independent both from the head pose and facial expressions of the subject.

1. regress sparse 3D facial key-points
2. estimate head pose
3. models the nonlinearity of deformations caused by facial expressions

obtain canonical keypoints:
$$
\mathbf{p}_k^{c a n}=\frac{1}{s} \mathbf{R}^{-1}\left(\mathbf{p}_k-\mathbf{d}_k-\mathbf{t}\right), \quad k=1, \ldots, K
$$

inverse:

$$
\mathbf{p}_k=s \mathbf{R} \mathbf{p}_k^{c a n}+\mathbf{t}+\mathbf{d}_k, \quad k=1, \ldots, K
$$

3d keypoints loss

loss between predict and target
$$
\mathcal{L}^p=\left\|\mathbf{p}^s-\mathbf{1}^s\right\|_2^2+\left\|\mathbf{p}^t-\mathbf{1}^t\right\|_2^2
$$

loss between reconstruction and target
$$
\mathcal{L}^{r e c}=\left\|\mathbf{p}^{s, r e c}-\mathbf{1}^s\right\|_2^2+\left\|\mathbf{p}^{t, r e c}-\mathbf{1}^t\right\|_2^2
$$

penalise the error between the predicted target head rotation
a regularisation term on the expression deformation vectors ensures that key-point perturbations due to expressions are kept small, as we want to avoid encoding identity-speciﬁc details in these vectors
$$
\mathcal{L}^R=\left\|\mathbf{R}^t-\mathbf{R}_*^t\right\|_2^2, \quad \mathcal{L}^d=\left\|\mathbf{d}^s\right\|_2^2+\left\|\mathbf{d}^t\right\|_2^2
$$

$$
\mathcal{L}_{E_{c a n}}=\lambda_p \mathcal{L}^p+\lambda_{\text {rec }} \mathcal{L}^{r e c}+\lambda_R \mathcal{L}^R+\lambda_d, \mathcal{L}^d
$$

### gaze estimation

- predict 3D meshes of eyes instead of 3D gaze vectors or angles
  estimating dense geometry instead of few pose parameters, has recently been shown to beneﬁt face and body pose estimation systems
- based on 2D sparse landmarks around the iris contour and the available gaze labels.

losses

predicted coordinates v and the corresponding pseudo-ground truth ones v* :
$$
\mathcal{L}^v=\left\|\mathbf{v}-\mathbf{v}^*\right\|_1
$$

between the 3N t eye edge lengths e computed from v and those, e ∗ , computed from v ∗ 
$$
\mathcal{L}^e=\left\|\mathbf{e}-\mathbf{e}^*\right\|_1
$$
gaze loss
$$
\mathcal{L}^g=(180 / \pi) \arccos \left(\mathbf{g}^T \mathbf{g}^*\right)
$$

$$
\mathcal{L}_{E_{\text {gaze }}}=\lambda_v \mathcal{L}^v+\lambda_e \mathcal{L}^e+\lambda_g \mathcal{L}^g
$$

$$
\theta=\arctan \sqrt{g_x^2+g_y^2} / g_z, \quad \phi=\arctan g_y / g_z
$$

### Images synthesis

![](https://markdown-imagebed.oss-cn-beijing.aliyuncs.com/imgs/202209142154324.png)

- based on the generator of HeadGAN
- draw a source and a target image sketch. Gaze angles are color coded in the sketches, within the areas deﬁned by keypoints that belong to the eyes.
- network G is comprised of two modules: a ﬂow network and a rendering network

1. the ﬂow network extracts visual feature maps from the source image and its corresponding key-point sketch in multiple spatial resolutions
2. target sketch is injected into the network, guides the prediction of the optical ﬂow w
3. we utilise the backward optical ﬂow warping operator from FlowNet ~`2.0
4. The rendering network passes the target sketch through an encoder and combines the extracted feature map with (i) the warped features and warped image $\overline{\mathbf{x}}=\mathbf{w}\left(\mathbf{x}^s\right)$

#### N-shot extension

- addition of one more output layer to the ﬂow network, which now learns to compute a set of 2D weights $m \in{\mathbb{R}^{H×W}}$

$$
\overline{\mathbf{x}}=\frac{\sum_j^M \exp \left(\mathbf{m}_j\right) \mathbf{w}_j\left(\mathbf{x}_j^S\right)}{\sum_j^M \exp \left(\mathbf{m}_j\right)}
$$

warped features:

$$
\overline{\mathbf{h}}^{(i)}=\frac{\sum_j^M \exp \left(\mathbf{m}_j\right) \mathbf{w}_j\left(\mathbf{h}_j^{(i)}\right)}{\sum_j^M \exp \left(\mathbf{m}_j\right)}, \quad i=1, \ldots, L
$$


- optical ﬂow and rendering networks that make G are trained jointly
- minimising the distance between feature maps extracted from multiple layers of a pre-trained VGG network
 $$
\mathcal{L}_G^{V G G}=\sum\left\|V G G_l(\tilde{\mathbf{x}})-V G G_l\left(\mathbf{x}^t\right)\right\|_1
$$

- we apply a VGG loss on the warped image, in order to force the ﬂow network to learn a correct ﬂow from the source image to the desired head pose, which gives the loss term $\mathcal{L}_F^{VGG}$
- improve the photo-realism of synthetic images by placing an adversarial loss term $\mathcal{L}^{adv}_G$
- speciﬁcally a Hinge GAN loss
- we employ two critics, a general image discriminator $D_I$ and a dedicated discriminator $D_M$ for the mouth area

$$
\mathcal{L}_G=\mathcal{L}_G^{a d v}+\lambda_{V G G}\left(\mathcal{L}_G^{V G G}+\mathcal{L}_F^{V G G}\right)+\lambda_{F M} \mathcal{L}_G^{F M}
$$

### Inference
1. we use $E_{canto}$ regress 3D key-points, head pose and expression deformations from $x_s$ obtain the canonical key-points $\mathbf{p}^{s, c a n}$
2. estimate the target pose and expression $\mathcal{T}^t, \mathbf{d}^t$ from $\mathbf{x}^t$, The adapted target key-points $\mathbf{p}^{s, a d a p t}$ are obtained with the application of Eq. 2
3. estimate eye gaze from both images with E gaze network and ﬁnally draw the sketches that serve as input to generator G