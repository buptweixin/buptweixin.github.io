---
title: "Pixel-in-Pixel Net: Towards Efficient Facial Landmark Detection in the Wild"
alias: jinPixelinPixelNetEfficient2021
tags:
  - landmark
  - heatmap
  - regression
rating: ⭐⭐⭐
shared: true
ptype: article
draft: false
date: 2022-12-27
---


# Pixel-in-Pixel Net: Towards Efficient Facial Landmark Detection in the Wild
<cite>* Authors: [[Haibo Jin]], [[Shengcai Liao]], [[Ling Shao]]</cite>

* DOI: [10.1007/s11263-021-01521-4](https://doi.org/10.1007/s11263-021-01521-4)

* [Local library](zotero://select/items/1_84T2L4XC)

***

### 初读印象

comment:: PIPNet 提出了一种粗粒度heatmap之后细粒度回归的关键点检测算法，结合了 heatmap方法高精度和回归方法高速度的优点，在多个bmk上取得了很好的效果。

### 文章骨架
%%创新点到底是什么?%%
novelty:: 结合了 heatmap 和 regression 的优点，达到又快又好的关键点检测效果

%%有什么意义？%%
significance:: 保证 heatmap 类方法高精度的同时保证了推理速度。

%%有什么潜力?%% 
potential:: 


### TL;DR 

当前主流的 landmark 检测方法主要有两类， 一类基于 heatmap 的，如图3(b) 所示，一类直接回归坐标，如图3(a)所示。 
heatmap 方法优点是准确性高，缺点是计算复杂度高、缺少全局约束（表现如被遮挡区域预测不受控制）， 而直接回归坐标的方法计算复杂度低且有比较好的全局形状约束，但是精度相对较低。

本文提出一种名为 PIPNet 的方法， 希望结合两类方法的优点。

### 计算复杂度问题

首先， 为了解决 heatmap 方法计算复杂度高的问题， PIPNet 取消了上采样过程，直接在下采样后的 $N\times{W_I/s}\times{H_I/s}$ s 为 stride，N 为landmark 数目, $W_I,H_I$ 分别为输入分辨率，比如输入 256x256 的数据，s = 32， 那么获得的 feature map 大小为 8x8，通过 feature map 上最大值定位当前关键点的粗糙位置，显然这个分辨率的 faeture map 只对应 64 个位置是不足以应对 landmark 检测任务的， 为此， 作者额外增加了一个 $2\times{N}$ 的分支 offset 预测分支， 这个分支的输出是以粗糙位置所在的 grid 的左上角为基准的精细位置。具体结构如图 3(c) 所示，而坐标形式的 gt 映射到新结构的过程如图 4 所示。

![](https://markdown-imagebed.oss-cn-beijing.aliyuncs.com/imgs20220213121548.png)

![](https://markdown-imagebed.oss-cn-beijing.aliyuncs.com/imgs20220213121612.png)


### 全局信息感知能力

如图 5 所示，当人脸角度很大时，PIPNet 和 feature map 预测一样会存在很大的偏差。这是因为坐标回归方法输出来源于 fc， 所有点的feature 能互相感知到， 但是 PIPNet 各个点是相互独立获取的，缺乏这样的全局信息。

![](https://markdown-imagebed.oss-cn-beijing.aliyuncs.com/imgs20220213165817.png)

为了提升模型的全局感知能力，在上面模型的基础上，作者提出了 neighbor regression module (NRM) 模块，在预测本身的 offset 以外， 这个分支还会预测当前点周围最近的 C 个点的 offset。 neighbors.

### 泛化能力问题

![](https://markdown-imagebed.oss-cn-beijing.aliyuncs.com/imgs20220213172012.png)

为了提升模型的泛化能力，提出了 self-training with curriculum (STC) 自学习模块， 和传统自学习一直针对一个任务进行不同， STC 会基于异源数据从难到易的三个任务学习。具体来说三个任务不同之处是 feature map 对应的 stride 不同， 也就是对应的分辨率不同，越大的 stride 对应的分辨率越低， 存在的负样本数量越少，也即任务越简单。 

具体步骤：
![](https://markdown-imagebed.oss-cn-beijing.aliyuncs.com/imgs/202203171103456.png)

1. 用人工标注的图片训练  PIPNet；
2. 用上一步得到的模型生成未标注数据的伪标签；
3. 使用标注和伪标签生成新的数据集；
4. 用人工数据集训练 task3， 新数据集训练 task 1- 2
重复 2-4 步骤


### References

1. [10.1109/BTAS.2017.8272731](https://doi.org/10.1109/BTAS.2017.8272731)
2. [10.1145/1553374.1553380](https://doi.org/10.1145/1553374.1553380)
3. [10.1109/ICCV.2013.191](https://doi.org/10.1109/ICCV.2013.191)
4. [10.1109/CVPR42600.2020.00590](https://doi.org/10.1109/CVPR42600.2020.00590)
5. [10.1007/978-3-319-46454-1_8](https://doi.org/10.1007/978-3-319-46454-1_8)
6. [[@chenFaceAlignmentKernel2019]]
7. [10.1109/CVPR.2018.00352](https://doi.org/10.1109/CVPR.2018.00352)
8. [[@dapognyDeCaFADeepConvolutional2019]]
9. [10.1007/s11263-018-1134-y](https://doi.org/10.1007/s11263-018-1134-y)
10. [10.1109/CVPR42600.2020.00525](https://doi.org/10.1109/CVPR42600.2020.00525)
11. [10.1109/CVPR.2018.00110](https://doi.org/10.1109/CVPR.2018.00110)
12. [10.1109/ICCV.2019.00087](https://doi.org/10.1109/ICCV.2019.00087)
13. [10.1109/CVPR.2018.00047](https://doi.org/10.1109/CVPR.2018.00047)
14. [10.1109/CVPR.2017.392](https://doi.org/10.1109/CVPR.2017.392)
15. [[@fengWingLossRobust2018]]
16. This reference does not have DOI 😵
17. This reference does not have DOI 😵
18. [10.1109/CVPR.2014.306](https://doi.org/10.1109/CVPR.2014.306)
19. [10.1109/CVPRW.2017.255](https://doi.org/10.1109/CVPRW.2017.255)
20. [10.1109/CVPR.2016.619](https://doi.org/10.1109/CVPR.2016.619)
21. [10.1109/CVPR.2018.00167](https://doi.org/10.1109/CVPR.2018.00167)
22. [10.1109/ICCV.2019.00140](https://doi.org/10.1109/ICCV.2019.00140)
23. This reference does not have DOI 😵
24. [10.1109/CVPR.2019.00503](https://doi.org/10.1109/CVPR.2019.00503)
25. [10.1109/ICCV.2017.409](https://doi.org/10.1109/ICCV.2017.409)
26. This reference does not have DOI 😵
27. [10.1109/ICCVW.2011.6130513](https://doi.org/10.1109/ICCVW.2011.6130513)
28. [[@kumarLUVLiFaceAlignment2020]]
29. [10.1109/TPAMI.2012.191](https://doi.org/10.1109/TPAMI.2012.191)
30. [10.1109/TPAMI.2017.2734779](https://doi.org/10.1109/TPAMI.2017.2734779)
31. [10.1109/CVPR.2017.713](https://doi.org/10.1109/CVPR.2017.713)
32. [10.1109/ICCV.2015.425](https://doi.org/10.1109/ICCV.2015.425)
33. [10.1109/CVPR.2019.00358](https://doi.org/10.1109/CVPR.2019.00358)
34. This reference does not have DOI 😵
35. [10.1109/CVPR.2017.393](https://doi.org/10.1109/CVPR.2017.393)
36. [10.1109/CVPR.2018.00088](https://doi.org/10.1109/CVPR.2018.00088)
37. [10.1007/978-3-319-46484-8_29](https://doi.org/10.1007/978-3-319-46484-8_29)
38. [10.1109/CVPR.2017.395](https://doi.org/10.1109/CVPR.2017.395)
39. [10.1007/978-3-030-01264-9_17](https://doi.org/10.1007/978-3-030-01264-9_17)
40. [10.1109/CVPR.2016.146](https://doi.org/10.1109/CVPR.2016.146)
41. [10.1109/ICCV.2019.01025](https://doi.org/10.1109/ICCV.2019.01025)
42. [10.1109/TIP.2016.2518867](https://doi.org/10.1109/TIP.2016.2518867)
43. [10.1109/ICCV.2019.01020](https://doi.org/10.1109/ICCV.2019.01020)
44. [10.1007/978-3-319-24574-4_28](https://doi.org/10.1007/978-3-319-24574-4_28)
45. [10.1109/ICCVW.2013.59](https://doi.org/10.1109/ICCVW.2013.59)
46. [10.1109/CVPR.2019.00712](https://doi.org/10.1109/CVPR.2019.00712)
47. [10.1109/CVPR.2018.00474](https://doi.org/10.1109/CVPR.2018.00474)
48. [10.1109/ICCVW.2015.132](https://doi.org/10.1109/ICCVW.2015.132)
49. [10.1109/CVPR.2013.446](https://doi.org/10.1109/CVPR.2013.446)
50. [[@taiHighlyAccurateStable2019]]
51. [10.1109/CVPR.2014.220](https://doi.org/10.1109/CVPR.2014.220)
52. [10.1007/978-3-030-01219-9_21](https://doi.org/10.1007/978-3-030-01219-9_21)
53. [10.1109/CVPR.2016.262](https://doi.org/10.1109/CVPR.2016.262)
54. [10.1109/CVPR.2016.453](https://doi.org/10.1109/CVPR.2016.453)
55. [[@valleDeeplyInitializedCoarsetofineEnsemble2018]]
56. [10.1016/j.cviu.2019.102846](https://doi.org/10.1016/j.cviu.2019.102846)
57. This reference does not have DOI 😵
58. [10.1109/ICCV.2019.00707](https://doi.org/10.1109/ICCV.2019.00707)
59. [10.1109/CVPR.2016.511](https://doi.org/10.1109/CVPR.2016.511)
60. [10.1109/CVPRW.2017.261](https://doi.org/10.1109/CVPRW.2017.261)
61. [[@wuLookBoundaryBoundaryAware2018b]]
62. [10.1007/978-3-030-01231-1_29](https://doi.org/10.1007/978-3-030-01231-1_29)
63. [10.1109/CVPRW.2017.253](https://doi.org/10.1109/CVPRW.2017.253)
64. [10.1109/CVPR.2016.596](https://doi.org/10.1109/CVPR.2016.596)
65. [10.1109/ICCV.2017.113](https://doi.org/10.1109/ICCV.2017.113)
66. [10.1109/CVPR.2019.00225](https://doi.org/10.1109/CVPR.2019.00225)
67. [10.1109/CVPRW.2017.263](https://doi.org/10.1109/CVPRW.2017.263)
68. [10.1109/TPAMI.2015.2469286](https://doi.org/10.1109/TPAMI.2015.2469286)
69. [10.1007/978-3-030-58621-8_31](https://doi.org/10.1007/978-3-030-58621-8_31)
70. [10.1007/978-3-030-01261-8_11](https://doi.org/10.1007/978-3-030-01261-8_11)
71. [10.1109/CVPR.2019.00360](https://doi.org/10.1109/CVPR.2019.00360)
72. This reference does not have DOI 😵
73. [10.1109/CVPR.2016.371](https://doi.org/10.1109/CVPR.2016.371)
74. [10.1109/CVPR.2019.00078](https://doi.org/10.1109/CVPR.2019.00078)
75. [10.1109/ICCV.2019.00023](https://doi.org/10.1109/ICCV.2019.00023)

 Currently 7 references inside library! @2022-12-28