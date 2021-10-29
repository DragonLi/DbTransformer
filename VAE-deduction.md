# 番茄鱼的机器学习的学习笔记
目录
- [数学知识](./数学知识.md)
- [VAE相关](./VAE相关.md)
- [极值理论](./极值理论.md)
### VAE聚类模型
VAE可以增加聚类的功能，如果提前知道有多少个类别，并且假设聚类只根据隐藏层来计算而跟输入无关，生成器（解码器）的输出也跟分类无关而只与隐藏层相关。也就是说聚类是一个独立正交的功能模块输入是$z$（或者等价于这种效果）。

假设分类用整数变量$y$表示，则隐藏层变量从$z$改为$(z,y)$，则优化目标函数为：

$$
\begin{split}
& min KL[p(x,z,y)//q(x,z,y)]  \\\\
& = min \sum_{y} \iint p(x,z,y) log\frac{p(x,z,y)}{q(x,z,y)}dzdx \\\\
&= min \sum_{y} \iint p(x)p(z,y/x)log\frac{p(x)p(z,y/x)}{q(x/z,y)q(z,y)}dzdx
\end{split}
$$

假设隐藏层的分类$y$只跟$z$有关而跟$x$无关（是设计目标之一），因此假设$p(y/z,x)=p(y/z)$。同理生成器的结果$x$只与$z$有关而跟$y$无关，因此假设$q(x/z,y)=q(x/z)$。

根据以上假设有：$p(z,y/x)=p(y/z,x)p(z/x)=p(y/z)p(z/x)$，$q(z,y)=q(z/y)q(y)$。优化目标简化为：
$$
\begin{split}
& min \sum_{y} \iint p(x)p(z,y/x)log\frac{p(x)p(z,y/x)}{q(x/z,y)q(z,y)}dzdx  \\\\
&= min \sum_{y} \iint p(x)p(y/z)p(z/x)log\frac{p(x)p(y/z)p(z/x)}{q(x/z)q(z/y)q(y)}dzdx
\end{split}
$$
与标准VAE一样，假设$p(z/x)$服从各分量独立的正态分布。然而$q(z/y)$依赖y因此服从标准正态分布$N(\mu_y,I)$，$q(y)$根据数据特征选择简单的离散概率分布（按照最大熵原理，没有其他信息的情况下可以选择均匀分布），$p(y/z)$是分类器，可以选择softmax，也可以增加其他网络加强分类器的能力。
如下所示：
$$
\begin{split}
x \sim p(x) 
\xrightarrow{Encoder} & z \sim p(z/x) \quad z \sim & q(z/y)
\xrightarrow{Decoder} x \sim q(x/z) \\\\
 & \downarrow & \uparrow \\\\
& p(y/z) & q(y)
\end{split}
$$

## 基于最优化方法的机器学习套路

### 背景：算法分类和最优化
分类和预测问题可以看成是拟合函数，而生成模型问题则认为是拟合分布。拟合分布有多种方法，VAE以重构为目标，GAN以鉴别为目标，另外还可以基于强化学习以及演化学习。

然而无论哪种方法，最终需要体现为定义合适的目标函数（最小化/最大化），使用泛函优化（概率分布是一个函数）的数学技巧，将难以优化的泛函优化问题变成可以优化的损失函数和神经网络。

要注意损失函数与需要优化的目标函数的区别。优化目标函数很多时候不能直接计算，需要转换，而损失函数必须可以计算。

因此机器学习的套路是：需求分析->目标函数->损失函数->网络设计->数据收集->调参优化

### 仔细分析需求确定优化的目标函数
例子：深度学习的互信息：无监督提取特征
https://kexue.fm/archives/6024

区分不同分布是目标，但重构数据不是目标，而且两者还不等价！

### 拟合分布的函数类型
神经网络是一个参数化的函数。这个函数的输入如果满足某个概率分布，则输出的值会满足另外一个分布。如果类型系统携带上概率分布的信息，可以作为文档和推理检验的有用信息。

具体表现：输入可以是满足某个分布的向量，而输出则是想计算得到的目标分布的向量。
$$
T: \\{ P: Probability,Q: Vec(R,n) \to Probability \\} \to (x:P) \to (y:Q(x))
$$
其中P和Q都是表示概率分布，生成代码时可以删除。

## 数据增强/提升框架
[参考]
(https://mp.weixin.qq.com/s?__biz=MzA5ODEzMjIyMA==&mid=2247653464&idx=1&sn=4cea82edf390d2b9b7feecba32733963&scene=58&subscene=0&exportkey=A%2FeVd4jwuypjJBkWq066NEs%3D&pass_ticket=xY8f5SA8zxXMB7wrWPl0IfIfkbG5f5ayTp8t4lf4funRXPdHy3mGaF2ZlfV76GQC&wx_header=0  "关注数据而不是模型：我是如何赢得吴恩达首届 Data-centric AI 竞赛的")

[英文连接][how-i-won-andrew-ngs-very-first-data-centric-ai-competition]

得奖的方法如下：

1. 通过训练数据得到生成模型，并生成一定数量的候选增强数据
2. 划分训练和验证数据集合，训练预测模型并预测验证数据的标签
3. 使用另外一个模型对图像数据（训练数据和增强数据）进行嵌入，得到类似词向量的模型 
4. 对每个预测错误的验证数据，利用图形嵌入向量使用余弦相似度从增强数据中查找临近点对应的增强数据，并加入到训练集合中
5. 重新训练预测模型，并预测验证集的标签
6. 重复4-6步，知道增强数据的数量达到上限

以上过程可以抽象为：

1. 针对数据特征确定生成-聚类模型，例如VAE聚类模型
2. 使用全部数据训练生成-聚类模型
3. 划分训练和验证数据集合，保证每个类别都有训练数据和验证数据
4. 训练预测模型，预测验证集标签
5. 对每个预测错误的数据，根据类别使用生成模型生成增强数据若干个，并加入到训练数据集合
6. 重复4-5步骤，知道满足终止条件（例如增强数据的数量达到上限，或则运行次数已经达到上限）

得奖方法把生成-聚类模型拆分为生成模型和对象嵌入模型，并使用Annoy算法包执行临近搜索。如果对象嵌入模型比较难设计，而数据分类的数量可以直接通过分析数据特征得到，通过VAE-Clustering算法模型可能更好，需要进一步探索。

[模型参考](https://kexue.fm/archives/5887 "变分自编码器（四）：一步到位的聚类方案")


[how-i-won-andrew-ngs-very-first-data-centric-ai-competition]: https://towardsdatascience.com/how-i-won-andrew-ngs-very-first-data-centric-ai-competition-e02001268bda  "how-i-won-andrew-ngs-very-first-data-centric-ai-competition"


## 可逆概率分布拟合模型 Normalized FLOW

## GAN原理

### f-GAN

### 对偶空间GAN

### 一般GAN

### GAN与VAE比较
虽然都可以通过变分推断推导，但是GAN假设的分布比VAE少。

## VAE目标函数分解为编码器部分以及解码器部分

### KL散度与ELBO

## 信息论知识

### 最小交叉熵原理

#### 特例：最大熵原理

#### 特例：EM算法
部分参考 https://kexue.fm/archives/5239

## 变分推断统一框架下的VAE/GAN/EM算法
部分参考 https://kexue.fm/archives/5716

都是概率估算在生成模型中的应用

## RNN系列与自动机

## GRU

$$
\begin{split}
reset_t & = \sigma(W^{rh} state_{t-1} + W^{rx} x_t + b^r) \quad reset \ old \ state \ in \ new \ cell   \\\\
update_t & = \sigma(W^{uh} state_{t-1} + W^{ux} x_t + b^u) \quad update  \ old \ state \ in \ new \ state \\\\
cell_t & = tanh(W^{ch} (state_{t-1} \circ reset_t) + W^{cx} x_t + b^c) \quad cell \ is \ candidate \\\\
state_t & = update_t \circ state_{t-1} + (1-update_t) \circ cell_t
\end{split}
$$

