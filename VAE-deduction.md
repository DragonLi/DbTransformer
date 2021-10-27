# 番茄鱼的机器学习的学习笔记

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

1. 针对数据特征确定生成-编码-聚类模型，例如VAE聚类模型
2. 使用全部数据训练生成-编码-聚类模型
3. 划分训练和验证数据集合，保证每个类别都有训练数据和验证数据
4. 训练预测模型，预测验证集标签
5. 对每个预测错误的数据，根据类别使用生成模型生成增强数据若干个，并加入到训练数据集合
6. 重复4-5步骤，知道满足终止条件（例如增强数据的数量达到上限，或则运行次数已经达到上限）

得奖方法把生成-编码-聚类模型拆分为生成模型和对象嵌入模型，并使用Annoy算法包执行临近搜索。然而如果要更通用，通过VAE-GAN-Clustering算法模型可能更好，需要进一步探索。

[模型参考](https://kexue.fm/archives/5887 "变分自编码器（四）：一步到位的聚类方案")


[how-i-won-andrew-ngs-very-first-data-centric-ai-competition]: https://towardsdatascience.com/how-i-won-andrew-ngs-very-first-data-centric-ai-competition-e02001268bda  "how-i-won-andrew-ngs-very-first-data-centric-ai-competition"


## 基于最优化方法的机器学习套路

### 背景：算法分类和最优化

### 仔细分析需求确定优化的目标函数
例子：深度学习的互信息：无监督提取特征
https://kexue.fm/archives/6024

区分不同分布是目标，但重构数据不是目标，而且两者还不等价！

### 拟合函数的类型
输入可以是满足某个分布的向量，而输出则是想计算得到的目标分布的向量。

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

## Extream Value Theory 极值理论
### 极值分布
几乎所有分布F(称为底分布)的极值（尾部）都可以被以下分布(CDF)渐进的逼近（O'Brien, George L. "Extreme values for stationary and Markov sequences." The Annals of Probability (1987): 281-291.）:
$$
G(z;\xi) = e^{-(1+\xi z)^{-\frac 1 \xi}} \quad \xi \neq 0, (1+\xi z)>0
$$

当$\xi > 0$时$z \in (-\frac 1 \xi,\infty)$时CDF用上式定义;当$z<-\frac 1 \xi$时CDF为0；当$z = -\frac 1 \xi$时取极限$\lim\limits_ { x \rightarrow 0 }x^y=0$。严格的定义如下($\xi > 0$)：
$$
G(z;\xi)=
\begin{cases}
0,& z\leq -\frac 1 \xi \\\\
e^{-(1+\xi z)^{-\frac 1 \xi}}, & z \in (-\frac 1 \xi,\infty)
\end{cases}$$
该分布称为$II$型极值分布。

当$\xi < 0$时情况有点不同（对偶）：
$$
G(z;\xi)=
\begin{cases}
e^{-(1+\xi z)^{-\frac 1 \xi}}, & z \in (-\infty,-\frac 1 \xi) \\\\
1,& z\geq -\frac 1 \xi
\end{cases}$$
该分布称为$III$型极值分布。

当$\xi = 0$时，通过取极限$\lim\limits_{\xi \rightarrow 0} [(1+\xi z)^{\frac 1 {\xi z}}]^{-z} = e^{-z} $得到(此时$(1+\xi z)=1>0$):
$$
G(z;\xi)=
e^{-e^{-z}}, \quad z \in (-\infty,+\infty)$$
该分布称为$I$型极值分布。


概率密度函数：
$$
pdf(z;\xi)=
\begin{cases}
e^{-z} \circ e^{-e^{-z}} ,& \xi = 0 \\\\
(1+\xi z)^{-(1 + \frac 1 \xi)} e^{-(1+\xi z)^{-\frac 1 \xi}}, & (1+\xi z) > 0
\end{cases}
$$

定义CDF取非平凡值的定义域为支撑集合，支撑集合的两个端点（上/下确界）分别为上/下(支撑)端点，使用数学符号记为：
$$
A= \\{ x: 0 < F(x) < 1 \\}=support(F), 
x_*=\inf\limits_{x \in A} A, 
x^{ * }=\sup\limits_{x \in A} A
$$

极值分布的支撑集合:
$$
A=
\begin{cases}
(-\infty,+\infty),& \xi=0,\quad I型 \\\\
(-\frac 1 \xi,+\infty),& \xi >0,\quad II型 \\\\
(-\infty,-\frac 1 \xi),& \xi < 0,\quad III型
\end{cases}
$$

为了书写方便，通常使用支撑集合内的CDF。

引入位置参数$\mu$和尺度参数$\sigma>0$，极值分布($z= \frac {x-\mu} \sigma$)可以泛化为(Generalized Extream Value Distribution)：
$$
GEV(x;\mu,\sigma,\xi) = e^{-(1+\xi \frac {x-\mu} \sigma)^{-\frac 1 \xi}} \quad , (1+\xi \frac {x-\mu} \sigma)>0
$$
上式中$\xi$为0的时候泛指取极限的情况。


### 极值分布的性质

$\xi < 1$时数学期望存在，其中$\gamma$是欧拉常数，$\Gamma$是整数上的阶乘函数在实数轴上的推广：
$$
E=
\begin{cases}
\mu+\sigma (\Gamma (1-\xi) - 1)/ \xi, & \xi \neq 0,\xi < 1 \\\\
\mu + \sigma \gamma, & \xi = 0 \\\\
\infty & \xi \geq 1
\end{cases}
$$

$\xi < 0.5$时方差存在：
$$
Var=
\begin{cases}
\sigma^2 (\Gamma (1-2\xi) - \Gamma^2 (1-\xi))/ \xi^2, & \xi \neq 0,\xi < 0.5 \\\\
\sigma^2 \frac {\pi^2} 6, & \xi = 0 \\\\
\infty & \xi \geq 0.5
\end{cases}
$$

信息熵：
$$
log(\sigma)+\gamma \xi + \gamma + 1
$$


### 超量分布 Peaks over Threshold
假设随机变量$X$服从CDF分布函数$F(X)$，则$z=(X-\mu)$是一个新的随机变量，并且可以定义以下超量分布的CDF函数：
$$
F_\mu(z) = Pr(X-\mu \leq z | X > \mu)=\frac{F(z+\mu)-F(\mu)}{1-F(\mu)},\quad z \geq 0
$$
称$F_\mu(z)$为随机变量$X$的超过阈值$\mu$的超出量的分布函数，简称超量分布。对应的密度函数为($f$是$F$的密度函数)：
$$
f_\mu(z)=\frac{f(z+\mu)}{1-F(\mu)}=\frac{f(z+\mu)}{\bar F(\mu)}
$$

### 超阈分布
类似于超量分布，也可以定义超阈分布：
$$
F_{[\mu]}(z) = Pr(X \leq z | X > \mu)=\frac{F(z)-F(\mu)}{1-F(\mu)},\quad z \geq \mu
$$
密度函数：
$$
f_{[\mu]}(z)=\frac{f(z)}{1-F(\mu)}=\frac{f(z)}{\bar F(\mu)}
$$

### 广义帕累托分布 Generalized Pareto distribution
超量分布在合适条件下可以被广义帕累托分布（Generalized Pareto distribution）所逼近。GP分布定义如下：
$$
GP(z;\xi)=
\begin{cases}
1 - (1+\xi z)^{-1 / \xi}, & \xi \neq 0,z \geq 0,(1+\xi z) \geq 0 \\\\
1 - e^{-z}, & \xi = 0,z \geq 0
\end{cases}
$$
概率密度函数:
$$
gp(z;\xi)=
\begin{cases}
(1+\xi z)^{-(1 + \frac 1 \xi)}, & \xi \neq 0,z \geq 0,(1+\xi z) \geq 0 \\\\
e^{-z}, & \xi = 0,z \geq 0
\end{cases}
$$
支撑集合:
$$
\begin{cases}
(0,+\infty), & \xi \geq 0 \\\\
(0,-1 / \xi], & \xi < 0
\end{cases}
$$

引入位置参数$\mu$和尺度参数$\sigma>0$，GP分布($z= \frac {x-\mu} \sigma$)可以泛化为：
$$
GP(x;\mu,\sigma,\xi) = 1-(1+\xi \frac {x-\mu} \sigma)^{-\frac 1 \xi} ,  \quad (1+\xi \frac {x-\mu} \sigma) \geq 0, \frac {x-\mu} \sigma \geq 0
$$
上式中$\xi$为0的时候泛指取极限的情况。

按照形状参数$\xi$可以分为三个类型：

$I型,\xi = 0$:

$$
GP_1(x;\mu,\sigma,\xi) =
\begin{cases}
0 , & x \leq \mu \\\\
1-e^{- \frac {x-\mu} \sigma}, & x > \mu
\end{cases}
$$

$II型,\xi > 0$:

$$
GP_2(x;\mu,\sigma,\xi) =
\begin{cases}
0 , & x \leq \mu \\\\
1-(1+\xi \frac {x-\mu} \sigma)^{-\frac 1 \xi}, & x > \mu
\end{cases}
$$

$III型,\xi < 0$:

$$
GP_3(x;\mu,\sigma,\xi) =
\begin{cases}
0, & x \leq \mu \\\\
1-(1+\xi \frac {x-\mu} \sigma)^{-\frac 1 \xi}, & \mu < x < \mu- \frac \sigma \xi \\\\
1, &  \mu- \frac \sigma \xi \leq x
\end{cases}
$$

概率密度函数泛化为：
$$
gp(x;\mu,\sigma,\xi)=
\begin{cases}
\frac 1 \sigma (1+\xi \frac {x-\mu} \sigma)^{-(1 + \frac 1 \xi)}, & \xi \neq 0,\frac {x-\mu} \sigma \geq 0,(1+\xi \frac {x-\mu} \sigma) \geq 0 \\\\
\frac 1 \sigma e^{-\frac {x-\mu} \sigma}, & \xi = 0,\frac {x-\mu} \sigma \geq 0
\end{cases}
$$
支撑集合:
$$
\begin{cases}
(\mu,+\infty), & \xi \geq 0 \\\\
(\mu,\mu- \frac \sigma  \xi], & \xi < 0
\end{cases}
$$

数学期望：
$$
E=
\begin{cases}
\mu+\frac {\sigma} {1-\xi}, & \xi < 1 \\\\
\infty , & \xi \geq 1
\end{cases}
$$

方差：
$$
Var=
\begin{cases}
\frac {\sigma^2} {(1-\xi)^2 (1-2 \xi)}, & \xi < 0.5 \\\\
\infty , & \xi \geq 0.5
\end{cases}
$$

#### 生成GP分布
如果随机变量$U$服从$(0,1]$上的均匀分布，则以下随机变量服从GP分布：
$$
\begin{split}
X &= \mu + \frac {\sigma (U^{-\xi} - 1)} {\xi} & \sim GP(X;\mu,\sigma,\xi) \quad & \xi \neq 0  \\\\
X &= \mu - \sigma ln(U) & \sim GP(X;\mu,\sigma,\xi) \quad & \xi = 0
\end{split}
$$


#### GP分布与GEV分布的关系
在支撑集合中有分布函数$GEV(X) > 0$，因此有
$$
GP(X;\mu,\sigma,\xi) - 1 = log (GEV(X;\mu,\sigma,\xi)) =
\begin{cases}
-(1+\xi \frac {x-\mu} \sigma)^{-\frac 1 \xi}, &\xi \neq 0 \\\\
-e^{- \frac {x-\mu} \sigma}, &\xi = 0
\end{cases}
$$
简单讲就是$GP(X)=1+log(GEV(X))$，因此GP分布很多性质与GEV分布类似。


### 超量分布逼近
如果随机变量$X$服从分布函数$F(X)=Pr(x \leq X)$，则称$\overline F(X) = Pr(x > X)=1- F(X)$为尾分布。

根据文献
* A. A. Balkema and L. De Haan. Residual life time at
great age. The Annals of probability, 1974
* J. Pickands III. Statistical inference using extreme
order statistics. the Annals of Statistics, 1975
* Extremes and Related Properties of Random Sequences and Process 1983

超量分布可以被GP分布逼近：给定独立同分布的随机变量样本序列$\\{ X_i \\}$，令$M_n=max \\{ X_i \\}$。如果存在规范化数列$\\{ a_n > 0 \\}$和$\\{  b_n \\}$，对于足够大的$n$满足
$$
Pr(\frac {M_n - b_n} {a_n} \leq x)\approx GEV(x;\mu,\sigma,\xi)
$$
则对于足够大的阈值$th$，随机变量$y=(X-th)$的分布服从GP分布：
$$
F(y)={GP}(y;0,\bar \sigma,\xi)=1-(1+ \frac {\xi y} {\bar \sigma}  )^{-1/\xi}
$$
其中$\bar \sigma=\sigma +\xi (th-\mu)$。
换个说法即是尾分布$\overline F(y) \sim (1+ \frac {\xi y} {\bar \sigma}  )^{-1/\xi}$


### GP分布参数估计方法
文献《实用极值统计方法》(史道济)列出了一些GP分布的参数估计方法。比较实用的方法是极大似然估计(Maximum likelihood estimation,MLE)，根据文献
* Anomaly detection in streams with extreme value theory
* Computing maximum likelihood estimates for the generalized pareto distribution

整理如下：

#### 对数似然函数和对应的微分似然方程
首先注意到超量分布的逼近中不需要考虑位置参数$\mu$，只需要考虑形状参数$\xi$和尺度参数$\sigma$。

其次注意随机变量的支撑集合是:
$$
\begin{cases}
(0,\infty),& \xi \geq 0 \\\\
(0, - \frac \sigma \xi], & \xi < 0
\end{cases}
$$

给定独立同分布的样本$\\{ X_i \\}$，似然函数是:
$$
L= \prod\limits_{k=1}^{n} \frac 1 \sigma (1+\xi \frac {x_i} \sigma)^{-(1 + \frac 1 \xi)}=\frac 1 {\sigma ^ n} \prod\limits_{k=1}^{n} (1+x_i \frac {\xi} \sigma)^{-(1 + \frac 1 \xi)}
$$
对数似然函数：
$$
l(\sigma,\xi)=log(L(\sigma,\xi))= -n log \sigma - (1 + \frac 1 \xi) \sum\limits_{k=1}^{n}log (1+x_i \frac {\xi} \sigma)
$$
如果$\xi = 0$，则对数似然函数为：
$$
\begin{split}
L & =\prod\limits_{k=1}^{n} \frac 1 \sigma e^{- x_i / \sigma} \\\\
l(\sigma,\xi)=log(L(\sigma)) & = -n log \sigma - \frac 1 \sigma \sum\limits_{k=1}^{n} x_i
\end{split}
$$

当$\xi < 0$时根据支撑集合的定义，$\\{ x_i \\}$需要满足$x_i \leq - \frac \sigma \xi$，记最大值$X_N=max \\{ x_i \\}$。如果$\xi < -1$，则$\lim\limits_{\frac \sigma \xi \rightarrow (-X_N)} log (1 + X_N \frac {\xi} \sigma) = - \infty $，此时$log(L(\sigma,\xi)) = + \infty$。因此$\xi < -1$时MLE无法使用。

当$\xi \neq 0$时,MLE的偏微分方程如下:
$$
\begin{split}
\frac {\partial l(\sigma,\xi)} {\partial \xi}
& = - (1 + \frac 1 \xi)'\sum\limits_{k=1}^{n}log (1+x_i \frac {\xi} \sigma) - (1 + \frac 1 \xi) \sum\limits_{k=1}^{n}log (1+x_i \frac {\xi}  \sigma)'  \\\\
& = \frac 1 {\xi ^ 2}\sum\limits_{k=1}^{n}log (1+x_i \frac {\xi} \sigma) - (1 + \frac 1 \xi) \sum\limits_{k=1}^{n} (1+x_i \frac {\xi}  \sigma)^{-1}(1+x_i \frac {\xi}  \sigma)'  \\\\
& = \frac 1 {\xi ^ 2} \sum\limits_{k=1}^{n}log (1+x_i \frac {\xi} \sigma) - (1 + \frac 1 \xi) \sum\limits_{k=1}^{n} (1+x_i \frac {\xi}  \sigma)^{-1} (\frac {x_i} \sigma)  \\\\
& = \frac 1 {\xi ^ 2} \sum\limits_{k=1}^{n}log (1+x_i \frac {\xi} \sigma) - \frac 1 \xi (1 + \frac 1 \xi) \sum\limits_{k=1}^{n} \frac {x_i \frac {\xi}  \sigma} {1+x_i \frac {\xi}  \sigma}    \quad (\xi \rightarrow 0 用到)\\\\
& = \frac 1 {\xi ^ 2} \sum\limits_{k=1}^{n}log (1+x_i \frac {\xi} \sigma) - \frac 1 \xi (1 + \frac 1 \xi) \sum\limits_{k=1}^{n}(1 - \frac 1 {1+x_i \frac {\xi}  \sigma})  \\\\
& = \frac 1 {\xi ^ 2} \sum\limits_{k=1}^{n}log (1+x_i \frac {\xi} \sigma) + \frac 1 \xi (1 + \frac 1 \xi) \sum\limits_{k=1}^{n} \frac 1 {1+x_i \frac {\xi}  \sigma}- \frac n \xi (1 + \frac 1 \xi)
\end{split}
$$
以及
$$
\begin{split}
\frac {\partial l(\sigma,\xi)} {\partial \sigma}
& = - \frac n \sigma - (1 + \frac 1 \xi) \sum\limits_{k=1}^{n}log (1+x_i \frac {\xi} \sigma)'  \\\\
& = - \frac n \sigma - (1 + \frac 1 \xi) \sum\limits_{k=1}^{n} \frac 1 {1+x_i \frac {\xi} \sigma}(1+x_i \frac {\xi} \sigma)'  \\\\
& = - \frac n \sigma - (1 + \frac 1 \xi) \sum\limits_{k=1}^{n} \frac 1 {1+x_i \frac {\xi} \sigma}(- \frac {x_i \xi} {\sigma^2} )  \\\\
& = - \frac n \sigma + \frac 1 \sigma (1 + \frac 1 \xi) \sum\limits_{k=1}^{n} \frac {x_i \frac {\xi} \sigma} {1+x_i \frac {\xi} \sigma}  \quad (\xi \rightarrow 0 用到) \\\\
& = - \frac n \sigma + \frac 1 \sigma (1 + \frac 1 \xi) \sum\limits_{k=1}^{n} (1 - \frac {1} {1+x_i \frac {\xi} \sigma})  \\\\
& = - \frac n \sigma + \frac n\sigma (1 + \frac 1 \xi) - \frac 1 \sigma (1 + \frac 1 \xi) \sum\limits_{k=1}^{n} \frac {1} {1+x_i \frac {\xi} \sigma}  \\\\
& = \frac n {\sigma \xi} - \frac 1 \sigma (1 + \frac 1 \xi) \sum\limits_{k=1}^{n} \frac {1} {1+x_i \frac {\xi} \sigma}
\end{split}
$$

以上偏微分方程组没有解析解，只能通过数值算法进行求解数值解。

因为：
$$
\begin{split}
&\lim\limits_{\xi \rightarrow 0}{} (1 + \frac 1 \xi) \sum\limits_{k=1}^{n} \frac {x_i \frac {\xi} \sigma} {1+x_i \frac {\xi} \sigma} \\\\
= & \lim\limits_{\xi \rightarrow 0}{} (1 + \frac 1 \xi) \sum\limits_{k=1}^{n} (\frac {1+x_i \frac {\xi} \sigma} {x_i \frac {\xi} \sigma})^{-1} \\\\
= & \lim\limits_{\xi \rightarrow 0}{} (1 + \frac 1 \xi) \sum\limits_{k=1}^{n} (1 + \frac {1} {\xi \frac {x_i} \sigma})^{-1}   \\\\
= & \lim\limits_{\xi \rightarrow 0}{} \sum\limits_{k=1}^{n} \frac {(1 + \frac 1 \xi)} {(1 + \frac {1} {\xi \frac {x_i} \sigma})}  \\\\
= & \lim\limits_{\xi \rightarrow 0}{} \sum\limits_{k=1}^{n} \frac {\xi + 1} {\xi + \frac {1} { \frac {x_i} \sigma}}  \\\\
= & \sum\limits_{k=1}^{n} \frac {1} {\frac {1} { \frac {x_i} \sigma}} = \frac 1 {\sigma} \sum\limits_{k=1}^{n}  x_i
\end{split}
$$
所以：
$$
\begin{split}
\lim\limits_{\xi \rightarrow 0}{} \frac {\partial l(\sigma,\xi)} {\partial \sigma} & = \frac 1 {\sigma ^2} \sum\limits_{k=1}^{n}  x_i - \frac n {\sigma}
\end{split}
$$

因为：
$$
\begin{split}
& \lim\limits_{x \rightarrow 0}{} [\frac 1 {x^2} log(1+ax) - (1 + \frac 1 x)(\frac a {1 + ax} )] \\\\
=& \lim\limits_{x \rightarrow 0}{}\frac {log(1+ax) - \frac {ax(x+1)} {1+ax} } {x^2} \\\\
(根据洛必达法则) = & \lim\limits_{x \rightarrow 0}{} \frac {(log(1+ax) - \frac {ax(x+1)} {1+ax})' } {(x^2)'} \\\\
= & \lim\limits_{x \rightarrow 0}{} \frac {\frac a {1+ax} - [(ax(x+1))'\frac 1 {1+ax} + ax(x+1)(\frac 1 {1+ax})']} {2x} \\\\
= & \lim\limits_{x \rightarrow 0}{} \frac {\frac a {1+ax} - [\frac {2ax+a} {1+ax} - ax(x+1)(\frac a {(1+ax)^2})]} {2x} \\\\
= & \lim\limits_{x \rightarrow 0}{} \frac {a(1+ax) - (2ax+a) (1+ax) + ax(x+1)a} {2x(1+ax)^2} \\\\
= & \lim\limits_{x \rightarrow 0}{} -\frac {a^2x^2-a^2x+2ax} {2x(1+ax)^2} \\\\
= & \lim\limits_{x \rightarrow 0}{} -\frac {a^2x-a^2+2a} {2(1+ax)^2} \\\\
= & -\frac {-a^2+2a} 2 = \frac {a^2} 2 - a
\end{split}
$$
令$a=\frac {x_i} \sigma$，$x=\xi$，有：
$$
\begin{split}
& \lim\limits_{\xi \rightarrow 0}{} \frac 1 {\xi ^ 2} log (1+ \frac {x_i} \sigma \xi) - (1 + \frac 1 \xi) \frac {\frac {x_i} \sigma} {1 + \frac {x_i} \sigma \xi} \\\\
= & \frac {(\frac {x_i} \sigma)^2} 2 - \frac {x_i} \sigma = \frac {x_i^2} {2 \sigma ^2} - \frac {x_i} \sigma
\end{split}
$$
所以
$$
\begin{split}
\lim\limits_{\xi \rightarrow 0}{} \frac {\partial l(\sigma,\xi)} {\partial \xi} & = \frac 1 {2 \sigma ^2} \sum\limits_{k=1}^{n}  x_i^2 - \frac 1 {\sigma} \sum\limits_{k=1}^{n}  x_i
\end{split}
$$

以上两个偏微分方程解存在的条件是$\overline{X^2}=2(\overline{X})^2$，而$Var(X)=E(X^2)-[E(X)]^2$，因此以上条件等价于样本方差等于样本均值的评分，刚好就是指数分布的特征（$\xi = 0$时GP分布就是特化为指数分布！）。因此以上条件不满足的情况下$\xi \neq 0$。

#### Grimshaw法极大似然估计
由上一节可知使用MLE方法评估参数时的搜索空间是$\\{ 0 > \xi \geq -1, \sigma > 0, - \frac \sigma \xi > max\\{ x_i \\}  \\} \bigcup \\{ \xi > 0, \sigma > 0 \\}$

搜索过程可以分成两部分，第一部分搜索在搜索空间内寻找局部最大值，第二部分是在边界$\xi = -1$上查找局部最大值。

令偏微分方程组等于零，变换得到：
$$
\begin{split}
n(1+\xi) & = \sum\limits_{i=1}^{n} log(1+\frac \xi \sigma x_i) + (1+\xi) \sum\limits_{i=1}^{n} (1+ \frac \xi \sigma x_i)^{-1} \\\\
n & =(1+\xi) \sum\limits_{i=1}^{n} (1+ \frac \xi \sigma x_i)^{-1}
\end{split}
$$
第二个等式代入第一个等式得到:
$$
\begin{split}
n(1+\xi) & = \sum\limits_{i=1}^{n} log(1+\frac \xi \sigma x_i) + n \\\\
n\xi & = \sum\limits_{i=1}^{n} log(1+\frac \xi \sigma x_i) \\\\
\xi & = \frac 1 n \sum\limits_{i=1}^{n} log(1+\frac \xi \sigma x_i)
\end{split}
$$
将上式重新代入第二个等式得到：
$$
\begin{split}
n & = [1+ \frac 1 n \sum\limits_{i=1}^{n} log(1+\frac \xi \sigma x_i)] [ \sum\limits_{i=1}^{n} (1+ \frac \xi \sigma x_i)^{-1} ] \\\\
1 & = [1+ \frac 1 n \sum\limits_{i=1}^{n} log(1+\frac \xi \sigma x_i)] [ \frac 1 n \sum\limits_{i=1}^{n} (1+ \frac \xi \sigma x_i)^{-1} ]
\end{split}
$$
令$\theta = \frac \xi \sigma$，以上等式总结为：
$$
\begin{split}
[1+ \frac 1 n \sum\limits_{i=1}^{n} log(1+\theta x_i)] [ \frac 1 n \sum\limits_{i=1}^{n} (1+ \theta x_i)^{-1} ] & = 1 \\\\
\frac 1 n \sum\limits_{i=1}^{n} log(1+\theta x_i) & = \xi
\end{split}
$$
由于第一个等式是约束比值$\theta =\frac \xi \sigma$，而第二个等式可以从比值直接计算出$\xi$，因此只需要按照第一个等式的约束搜索比值即可。

重写似然函数为：
$$
\begin{split}
& l(\theta) \\\\
= & -n log \sigma - (1 + \frac 1 \xi) \sum\limits_{k=1}^{n}log (1+x_i \frac {\xi} \sigma) \\\\
= & -n log (\frac \xi \theta) - (1 + \frac 1 \xi) \sum\limits_{k=1}^{n}log (1+ \theta x_i) \\\\
= & n log \theta - n log \xi - (1 + \frac 1 \xi) \sum\limits_{k=1}^{n}log (1+ \theta x_i) \\\\
= & n log \theta - n log (\frac 1 n \sum\limits_{i=1}^{n} log(1+\theta x_i)) - (1 + \frac 1 \xi) \sum\limits_{k=1}^{n}log (1+ \theta x_i) \\\\
= & n log \theta - n log (\frac 1 n \sum\limits_{i=1}^{n} log(1+\theta x_i)) - (1 + \frac 1 \xi) n \xi \\\\
= & n log \theta - n log (\frac 1 n \sum\limits_{i=1}^{n} log(1+\theta x_i)) - n - n \xi \\\\
= & n log \theta - n - n log (\frac 1 n \sum\limits_{i=1}^{n} log(1+\theta x_i)) - \sum\limits_{k=1}^{n}log (1+ \theta x_i)
\end{split}
$$

记$S=\sum\limits_{i=1}^{n} (1+ \theta x_i)^{-1}$，$L=\sum\limits_{i=1}^{n} log(1+\theta x_i)$，有
$$
\begin{split}
L' &=\sum\limits_{k=1}^{n} x_i (1+ \theta x_i)^{-1}=\frac 1 \theta \sum\limits_{k=1}^{n} x_i \theta(1+ \theta x_i)^{-1} \\\\
&=\frac 1 \theta \sum\limits_{k=1}^{n} [1-(1+ \theta x_i)^{-1}] \\\\
&=\frac n \theta - \frac 1 \theta\sum\limits_{k=1}^{n} (1+ \theta x_i)^{-1} = \frac {n-S} \theta \\\\
l(\theta)&=n log \theta - n - n log (\frac 1 n L) - L
\end{split}
$$

根据概率密度的定义必须满足$1+\theta x_i > 0$，即是$\theta > max \\{ -{x_i}^{-1} \\} =-min \\{ {x_i}^{-1} \\} = - \frac 1 {max \\{ x_i \\} } = - X_N^{-1}$，其中根据支撑集合$x_i > 0$。如果$\theta > 0$则必然满足上述条件。

由于$\theta \rightarrow (-\frac 1 {X_N})^+$时$\lim\limits_{\frac \sigma \xi \rightarrow (-X_N)} log (1 + X_N \frac {\xi} \sigma) = - \infty $，此时$log(L(\sigma,\xi)) = + \infty$。因此数值搜索必须避开下届$(-\frac 1 {X_N})$。也需要避开$0$，因为当且仅当为指数分布时才取零这个值。

更重要的一点是局部极大值与边界的无穷大之间由于函数连续性导致了必然存在对应的局部极小值。这意味着可能存在多个局部极大值，也就是说微分方程等于零的解可以有多个，增加了数值求解的难度（如果使用神经网络常规的手段梯度下降法搜索，则必须考虑到不同极小值的问题，导致初始化很难，或者需要多次搜索才能达到比较好的效果）。

重写的似然函数对应的偏微分为：
$$
\begin{split}
& \frac {\partial l(\theta)} {\partial \theta} \\\\
= & [n log \theta - n - n log (\frac 1 n L) - L]' \\\\
= & \frac n \theta - [n log (\frac 1 n L)]' - L' \\\\
= & \frac n \theta - (\frac n {\frac n L})L'/n - L' \\\\
= & \frac n \theta - (\frac n {L})L' - L' \\\\
= & \frac n \theta - (\frac n {L} + 1)\frac {n-S} \theta
\end{split}
$$

令$\frac {\partial l(\theta)} {\partial \theta}=0$，得到：
$$
\begin{split}
n &= (\frac n L + 1)(n-S) \\\\
nL & = (n+L) (n - S) \\\\
nL & = n^2 + nL - nS - LS \\\\
0 & = n^2-nS-LS \\\\
n^2 & = nS + LS \\\\
1 & = (1 + L/n)(S/n)
\end{split}
$$
等价于前面推导过的：
$$
\begin{split}
[1+ \frac 1 n \sum\limits_{i=1}^{n} log(1+\theta x_i)] [ \frac 1 n \sum\limits_{i=1}^{n} (1+ \theta x_i)^{-1} ] & = 1
\end{split}
$$

记
$$
h(\theta)=[1+ \frac 1 n \sum\limits_{i=1}^{n} log(1+\theta x_i)] [ \frac 1 n \sum\limits_{i=1}^{n} (1+ \theta x_i)^{-1} ] - 1=(1+ \frac 1 n L)(\frac 1 n S) - 1
$$

Grimshaw证明了以下定理：

1) $\lim\limits_{\theta \rightarrow (-1/X_N)^+} h(\theta)= -\infty$
2) 当$\theta > \theta_U = \frac {2[\overline{X} - X_{(1)}]} {[X_{(1)}]^2} $时$h(\theta) < 0$，其中$X_{(1)}$是样本最小值
3) 记$U=\sum\limits_{i=1}^{n} x_i (1+ \theta x_i)^{-2}$，$h'(\theta)= \frac S {n \theta} - \frac {S^2} {n^2 \theta} - \frac 1 n(1+ \frac L n ) U$
4) $\lim\limits_{\theta \rightarrow 0} h'(\theta) = 0$
5) $\lim\limits_{\theta \rightarrow 0} {h'}'(\theta) = \overline{X^2} - 2(\overline{X})^2$

证明：

1) $\lim\limits_{\theta \rightarrow (-1/X_N)^+} log(1+\theta X_N) = -\infty$，故$L \rightarrow -\infty$。而根据支持集合有$S>0$，因此$\lim\limits_{\theta \rightarrow (-1/X_N)^+} h(\theta)= -\infty$
2) 当$\theta > 0$时$log(1+\theta x)$是上凸函数，根据Jensen不等式有
$$
\frac 1 n \sum\limits_{i=1}^{n} log(1+\theta x_i) \leq log(1+\theta \overline{X})
$$
又因为$X_{(1)} \leq x_i$，因此${(1+\theta x_i)}^{-1} \leq {(1+\theta X_{(1)})}^{-1}$，从而
$$
\frac 1 n \sum\limits_{i=1}^{n} (1+ \theta x_i)^{-1} \leq {(1+\theta X_{(1)})}^{-1}
$$
根据以上两个不等式得到
$$
h(\theta) \leq [1+log(1+\theta \overline{X})] {(1+\theta X_{(1)})}^{-1}-1
$$
对$e^{\theta x}$在$x=0$和$\Delta x = X_{(1)}$使用泰勒展开
$$
e^{\theta X_{(1)}} = \sum\limits_{n=0}^{k} \frac {\theta^{n} e^{\theta X_{(1)}}} {n!} {X_{(1)}}^n + \frac {\theta^{k+1} e^{\theta \epsilon}} {(k+1)!} {X_{(1)}}^{k+1} > 1 + \theta {X_{(1)}} + \frac {(\theta {X_{(1)}})^2} 2
$$
如果需要$e^{\theta X_{(1)}}>1+\theta \overline{X}$，可以令
$$
\theta {X_{(1)}} + \frac {(\theta {X_{(1)}})^2} 2 > \theta \overline{X}
$$
即是$\theta > \theta_U = \frac {2[\overline{X} - X_{(1)}]} {[X_{(1)}]^2} $。此时$h(\theta) < 0$
3) 计算$h(\theta)$的导数时使用之前定义的符号$S$和$L$能够简化书写：
$$
\begin{split}
& h'(\theta) \\\\
= & \\{ [1+ \frac 1 n L] [ \frac 1 n S ] \\}' \\\\
= & (1+ \frac L n )' \frac S n + (1+ \frac L n )(\frac S n)' \\\\
= & \frac 1 n(\frac n \theta - \frac S \theta)\frac S n + (1+ \frac L n )(\frac S n)' \\\\
= & \frac S {n \theta} - \frac {S^2} {n^2 \theta} - \frac 1 n(1+ \frac L n ) U
\end{split}
$$
其中
$$
\begin{split}
S'= & [\sum\limits_{i=1}^{n} (1+ \theta x_i)^{-1}]' \\\\
= & - \sum\limits_{i=1}^{n} x_i (1+ \theta x_i)^{-2}=-U
\end{split}
$$
4) 根据定义
$$
\begin{split}
\lim\limits_{\theta \rightarrow 0} S = & \sum\limits_{i=1}^{n} (1+ 0 x_i)^{-1} = n \\\\
\lim\limits_{\theta \rightarrow 0} S' = & - \sum\limits_{i=1}^{n} x_i (1+ 0 x_i)^{-2} = - \sum\limits_{i=1}^{n} x_i = -n \overline{X} \\\\
\lim\limits_{\theta \rightarrow 0} U = & -\lim\limits_{\theta \rightarrow 0} S' = n \overline{X} \\\\
\lim\limits_{\theta \rightarrow 0} L = & \sum\limits_{i=1}^{n} log(1+0 x_i)=0
\end{split}
$$
根据洛必达法则
$$
\begin{split}
& \lim\limits_{\theta \rightarrow 0} h'(\theta) \\\\
= & \lim\limits_{\theta \rightarrow 0} [ \frac S {n \theta} - \frac {S^2} {n^2 \theta} - \frac 1 n(1+ \frac L n ) U ] \\\\
= & \lim\limits_{\theta \rightarrow 0} \frac 1 {n^2 \theta} (n S -S^2) - \overline{X} \\\\
= & \lim\limits_{\theta \rightarrow 0} \frac 1 {n^2} (n S' -2SS') - \overline{X} \\\\
= & \lim\limits_{\theta \rightarrow 0} \frac {S'} {n^2} (n  -2S) - \overline{X} \\\\
= & \frac {-n \overline{X}} {n^2} (n  -2n) - \overline{X} =0
\end{split}
$$
5) 根据定义
$$
\begin{split}
U'= & -2 \sum\limits_{i=1}^{n} {x_i^2} (1+ \theta x_i)^{-3} \\\\
\lim\limits_{\theta \rightarrow 0} U' = & -2 \sum\limits_{i=1}^{n} {x_i^2} (1+ 0 x_i)^{-3} = -2 \sum\limits_{i=1}^{n} {x_i^2}=-2 n\overline{X^2}
\end{split}
$$
计算${h'}'(\theta)$:
$$
\begin{split}
& {h'}'(\theta) \\\\
= & [\frac S {n \theta} - \frac {S^2} {n^2 \theta} - \frac 1 n(1+ \frac L n ) U]' \\\\
= & [\frac {n S -S^2} {n^2 \theta}]' - [\frac 1 n(1+ \frac L n ) U]' \\\\
= & \frac {(n S -S^2)'} {n^2 \theta}+ (n S -S^2) (\frac 1 {n^2 \theta })' - \frac 1 n [(1+ \frac L n )' U +(1+ \frac L n ) U'] \\\\
= & \frac {(n S' -2SS')} {n^2 \theta} - (n S -S^2) (\frac {(n^2 \theta)'} {n^4 \theta^2 }) - \frac 1 n[\frac {n-S} {n\theta} U +(1+ \frac L n ) U'] \\\\
= & \frac {(2SU - n U)} {n^2 \theta} + (S^2 - n S) (\frac {n^2} {n^4 \theta^2 }) - \frac 1 n[\frac {n-S} {n\theta} U +(1+ \frac L n ) U'] \\\\
= & \frac {(2SU - n U)} {n^2 \theta} + \frac {S^2 - n S} {n^2 \theta^2 } + \frac {SU-nU} {n^2 \theta} - \frac 1 n(1+ \frac L n ) U' \\\\
= & \frac { S^2 -nS + \theta(2SU-nU)+ \theta (SU-nU) } {n^2 \theta^2} - \frac 1 n[(1+ \frac L n ) U'] \\\\
= & \frac { S^2 -nS + 3\theta SU - 2n U \theta } {n^2 \theta^2} - \frac 1 n[(1+ \frac L n ) U']
\end{split}
$$
还有：
$$
\begin{split}
(SU)' & = S'U+SU' = SU' - U^2 \\\\
(\theta SU)' & = \theta' (SU) + \theta (SU)' = SU + \theta (SU' - U^2)\\\\
(\theta U)' & = \theta' U + \theta U' = U + \theta U'
\end{split}
$$
根据洛必达法则
$$
\begin{split}
& \lim\limits_{\theta \rightarrow 0} {h'}'(\theta) \\\\
= & \lim\limits_{\theta \rightarrow 0} \\{ \frac { S^2 -nS + 3\theta SU - 2n U \theta } {n^2 \theta^2} - \frac 1 n[(1+ \frac L n ) U'] \\} \\\\
= & \lim\limits_{\theta \rightarrow 0}  \frac { S^2 -nS + 3\theta SU - 2n U \theta } {n^2 \theta^2} - \lim\limits_{\theta \rightarrow 0} \frac 1 n[(1+ \frac L n ) U'] \\\\
= & \lim\limits_{\theta \rightarrow 0} \frac { S^2 -nS + 3\theta SU - 2n U \theta } {n^2 \theta^2} - \frac 1 n \lim\limits_{\theta \rightarrow 0} U' \\\\
= & \lim\limits_{\theta \rightarrow 0} \frac { 2SS' -nS' + 3(\theta SU)' - 2n (U \theta)'  } {2 n^2 \theta} + 2 \overline{X^2} \\\\
= & \lim\limits_{\theta \rightarrow 0} \frac { nU - 2SU  + 3(SU + \theta (SU' - U^2)) - 2n (U + \theta U')  } {2 n^2 \theta} + 2 \overline{X^2} \\\\
= & \lim\limits_{\theta \rightarrow 0} \frac { nU - 2SU  + 3SU - 2nU + 3\theta (SU' - U^2) - 2n \theta U'  } {2 n^2 \theta} + 2 \overline{X^2} \\\\
= & \lim\limits_{\theta \rightarrow 0} \frac { SU - nU } {2 n^2 \theta} + \lim\limits_{\theta \rightarrow 0} \frac { 3 (SU' - U^2) - 2n  U'  } {2 n^2} + 2 \overline{X^2} \\\\
= & \lim\limits_{\theta \rightarrow 0} \frac { SU - nU} {2 n^2 \theta} +  \frac { 3 (n(-2 n\overline{X^2}) - (n \overline{X})^2) - 2n (-2 n\overline{X^2})} {2 n^2} + 2 \overline{X^2} \\\\
= & \lim\limits_{\theta \rightarrow 0} \frac { SU - nU } {2 n^2 \theta} +  \frac { -2 n^2 \overline{X^2} -3n^2 (\overline{X})^2} {2 n^2} + 2 \overline{X^2} \\\\
= & \lim\limits_{\theta \rightarrow 0} \frac {  SU - nU } {2 n^2 \theta} -   {\overline{X^2} -1.5 (\overline{X})^2} + 2 \overline{X^2} \\\\
= & \lim\limits_{\theta \rightarrow 0} \frac { (SU)' - nU' } {2 n^2 } -1.5 (\overline{X})^2 + \overline{X^2} \\\\
= & \lim\limits_{\theta \rightarrow 0} \frac { (SU'-U^2) - nU' } {2 n^2 } -1.5 (\overline{X})^2 + \overline{X^2} \\\\
= & \frac { (n(-2 n\overline{X^2})-(n \overline{X})^2) - n(-2 n\overline{X^2}) } {2 n^2 } -1.5 (\overline{X})^2 + \overline{X^2} \\\\
= & \frac { -(n \overline{X})^2  } {2 n^2 } -1.5 (\overline{X})^2 + \overline{X^2} \\\\
= & \overline{X^2} -2(\overline{X})^2
\end{split}
$$

根据上述定义$\theta$的下界是$\theta_L=(-\frac 1 {X_N})$，上界是$\theta_U=\frac {2[\overline{X} - X_{(1)}]} {[X_{(1)}]^2}$，并且需要剔除零点（边界点上以及以外的函数值是负数或者无定义）。
根据上下界可以使用Newton-Raphson方程根搜索算法（牛顿-拉弗森迭代方法）。搜索方法是一种基于梯度的搜索方法，因此需要计算$h'(\theta)$。
二阶导数可以帮助算法对搜索区域零根个数进行判断。

* 当$\lim\limits_{\theta \rightarrow 0}{h'}'(\theta)>0$。由于$\lim\limits_{\theta \rightarrow 0^-}h'(\theta) = 0$($\theta<0$)，给定一个很小的$\epsilon>0$，根据二阶导数的定义有$h'(\theta-\epsilon) < h'(\theta) = 0$。又因为$h(\theta) = 0$，根据一阶导数的定义有$h(\theta-\epsilon) > h(\theta) =0$。加上$h(\theta_L)=-\infty$，因此$(\theta_L,\theta-\epsilon<0)$区间内连续函数$h(\theta)$穿过$x$轴的次数必须是奇数次。因为连续函数每次穿过$x$轴将改变函数值的符号一次。对于$\theta \rightarrow 0^+$的情况也是类似的。事实上由于$sign[h(\theta_L)]=sign[\theta_U]$，穿过$x$轴的次数是必然是偶数次，因此$(0,\theta_U)$的零根必然是奇数个。
* 当$\lim\limits_{\theta \rightarrow 0}{h'}'(\theta)<0$。分析过程类似，此时$(\theta_L,0)$和$(0,\theta_U]$都包含偶数个零根（可以是0个！）。

牛顿迭代法可以通过二分法获取初始解。二分法是指给定$h(a)*h(b) < 0$和足够小的$\epsilon > 0$，寻找一个值$c \in (a,b) $，令$|h(c)| < \epsilon$。迭代过程中$c=(a+b)/2$。

牛顿迭代法基本的迭代公式是：
$$
x_{n+1}=x_n - \frac {h(x_n)} {h'(x_n)}
$$
即是通过一阶（线性）逼近来获得更好的近似0根。
迭代停止条件为：
$$
| h(x_n) | < \epsilon
$$
迭代次数可以预先指定以免遇到不良函数时无法停止。

也可以通过改进的方式进行迭代，例如替换为$g(x) = h(x)/h'(x)$，$g(x)$与$h(x)$有相同的0根($h'(x) \neq 0$)，此时迭代需要用到$h(x)$的二阶导数：
$$
\begin{split}
g'(x) & = 1 - h(x)/[h'(x)]^2 {h'}'(x) \\\\
x_{n+1} &= x_n - \frac {g(x_n)} {g'(x_n)} = x_n - \frac {h(x_n) h'(x_n)} {h'(x_n)^2 - h(x_n) {h'}'(x_n)}
\end{split}
$$
得到0根$\theta^{(0)}$代入原方程可以得到候选的评估参数:
$$
\begin{split}
\xi & = \frac 1 n \sum\limits_{i=1}^{n} log(1+\theta^{(0)} x_i) \\\\
\sigma & = \frac \xi {\theta^{(0)}}
\end{split}
$$

注意边界$\xi = 1$时，似然函数在$\sigma \rightarrow X_N^+$时取极小值$-nlog\sigma$。因此MLE参数估计需要考虑所有$h(\theta)$的0根和$\xi = 1,\sigma=X_N$，哪个候选解的似然函数取得最大值哪个就是最终的参数评估值。

特别地，当$\overline{X^2}-2(\overline{X})^2=0$时可能是一个指数分布，此时$\theta = 0$并且似然函数需要按照指数分布来计算。

#### Siffer等人对GP分布参数评估MLE的改进
Alban Siffer, Pierre-Alain Fouque, Alexandre Termier, Christine Largouët等人利用对数不等式证明了$h(theta)>0$的下界，减少了搜索区域。

记$\theta_l =2 \frac { \overline{X}-X_{(1)} }  { \overline{X} X_{(1)} }$。假设$\theta > 0$和$\\{ x_i \\} > 0$($x_i$服从超量分布时一定满足)。Siffer等人证明了如果$\theta < \theta_l$，则$h(\theta)>0$。

当$\theta > 0$时$(1+\theta x_i)^{-1}$是下凸函数，根据Jensen不等式有
$$
\frac 1 n \sum\limits_{i=1}^{n} (1+\theta x_i)^{-1} \geq (1+\theta \overline{X})^{-1}
$$
根据对数不等式(令$b=1+x,a=1$且$x>0$):
$$
\begin{split}
(1+x) - 1 \leq (ln(1+x)-ln1) \cdot \frac {(1+x) +1} 2 \\\\
x \leq \frac {x+2} 2 ln(1+x) \\\\
ln(1+x) \geq \frac {2x} {x+2} = 2 - \frac 4 {2+x}
\end{split}
$$
又因为$X_{(1)} \leq x_i$，因此
$$
\begin{split}
\frac 1 n \sum\limits_{i=1}^{n} log(1+ \theta x_i) \geq \frac 1 n \sum\limits_{i=1}^{n} (2 - \frac 4 {2 + \theta x_i}) \\\\
\geq \frac 1 n \sum\limits_{i=1}^{n} (2 - \frac 4 {2 + \theta X_{(1)}}) =2-\frac 4 {2 + \theta X_{(1)}}
\end{split}
$$
根据以上两个不等式得到
$$
h(\theta) \geq (3-\frac 4 {2 + \theta X_{(1)}} ) (1+\theta \overline{X})^{-1}-1
$$
令
$$
(3-\frac 4 {2 + \theta X_{(1)}} ) (1+\theta \overline{X})^{-1}-1>0
$$则
$$
\begin{split}
&(3-\frac 4 {2 + \theta X_{(1)}} ) (1+\theta \overline{X})^{-1}-1 \\\\
= & \frac {3(2 + \theta X_{(1)}) - 4-(2 + \theta X_{(1)})(1+\theta \overline{X})} {(2 + \theta X_{(1)})(1+\theta \overline{X})} \\\\
= & \frac {6 + 3\theta X_{(1)} - 4-(2+\theta X_{(1)}+2\theta \overline{X}+\theta^2 X_{(1)} \overline{X})} {(2 + \theta X_{(1)})(1+\theta \overline{X})} \\\\
= & \frac {2\theta X_{(1)} -(2\theta \overline{X}+\theta^2 X_{(1)} \overline{X})} {(2 + \theta X_{(1)})(1+\theta \overline{X})} \\\\
= & \theta \frac {2 (X_{(1)} - \overline{X}) - \theta X_{(1)} \overline{X}} {(2 + \theta X_{(1)})(1+\theta \overline{X})} > 0
\end{split}
$$
因此当$\theta < 2 \frac { \overline{X}-X_{(1)} }  { \overline{X} X_{(1)} } = \theta_l$的时候$h(\theta) > 0$


根据以上定理，超量分布MLE评估的搜索空间可以改进为$(\theta_L,0)$和$[\theta_l,\theta_U]$

Sniffer等人没有使用方程求解工具计算$h(\theta)=0$的根，而是通过优化器求解以下函数的最小值：
$$
\sum\limits_{i=1}^{k} h(w_i)^2
$$
其中$\\{ w_i \\}$的初始值均匀分布于搜索空间$[\theta_L+\epsilon,-\epsilon]$和$[\theta_l,\theta_U]$，$\epsilon$是一个很小的数($10^{-8}/\overline{X}$，因为中间过程需要计算$\theta x_i$)，$k$取一个较小的数字(=10)，稍微大于估计的根个数。
当优化器返回另以上函数取最小值的参数$w_i^*$时，有较大可能包含全部$h(\theta)=0$的根。

Grimshaw使用Newton迭代法求方程解$h(\theta)=0$的根。搜索过程中利用二阶导数得知解的奇偶性，偶数时在两个区间内交替搜索，直到搜索到区间的边缘上停止。比起Sniffer等人的方法要麻烦很多，需要计算一阶和二阶导数。


### 最小值对偶相关结论
TODO



## GRU

$$
\begin{split}
reset_t & = \sigma(W^{rh} state_{t-1} + W^{rx} x_t + b^r) \quad reset \ old \ state \ in \ new \ cell   \\\\
update_t & = \sigma(W^{uh} state_{t-1} + W^{ux} x_t + b^u) \quad update  \ old \ state \ in \ new \ state \\\\
cell_t & = tanh(W^{ch} (state_{t-1} \circ reset_t) + W^{cx} x_t + b^c) \quad cell \ is \ candidate \\\\
state_t & = update_t \circ state_{t-1} + (1-update_t) \circ cell_t
\end{split}
$$

## VAE核心推导过程

$$
x 
\xrightarrow{p(z/x)} z
\xrightarrow{q(x/z)} x
$$
其中：p是编码器，q是解码器（生成器），z是隐藏变量

推导目标是生成模型的概率$q(x,z)$尽可能逼近真实概率$p(x,z)$

$$
\begin{split}
& min KL[p(x,z)//q(x,z)]  \\\\
& = min \iint p(x,z) log\frac{p(x,z)}{q(x,z)}dzdx \\\\
&= min \iint p(x)p(z/x)log\frac{p(x)p(z/x)}{q(x,z)}dzdx \\\\
&= min \iint p(x)p(z/x)\\{log[p(x)] + log\frac{p(z/x)}{q(x,z)} \\}dzdx \\\\
&= min\\{ \iint p(x)p(z/x)log[p(x)]dzdx + \iint p(x)p(z/x)log\frac{p(z/x)}{q(x,z)}dzdx\\} \\\\
&= min\\{ \int p(x)log[p(x)] (\int p(z/x)dz)dx + \int p(x) (\int p(z/x)log\frac{p(z/x)}{q(x/z)q(z)}dz)dx  \\} \\\\
&= min \\{ \int p(x)log[p(x)]dx+ \int p(x) (\int p(z/x) [log\frac{p(z/x)}{q(z)} - log(q(x/z))]dz) dx \\} \\\\
&= min \\{ -H(p(x)+ \int p(x) (KL[p(z/x)//q(z)]- \int p(z/x) log[q(x/z)]dz)dx \\} \\\\
& H(p(x)是服从p(x)密度的随机变量x的信息熵，是与隐变量z无关的常量 \\\\
&= min \int p(x) \\{ KL[p(z/x)//q(z)]- \int log[q(x/z)] p(z/x) dz \\} dx  \\\\
& \int log[q(x/z)] p(z/x) dz 不可以进一步归约为log[q(x/z)]，因为log[q(x/z)]是依赖变量z的 \\\\
&= min E_{x \sim p(x) } \\{ KL[p(z/x)//q(z)] - E_{z \sim p(z/x)} log[q(x/z)] \\} \cdots (1)
\end{split} 
$$

因此损失函数定义为$E_{x \sim p(x) } L$，而:
$$
L = KL[p(z/x)//q(z)] + E_{z \sim p(z/x)}( -log[q(x/z)] ) \cdots (2)
$$

变分下界ELBO定义为L的相反数，即是：
$$
ELBO[p(z/x),q(x/z),q(z)] = E_{z \sim p(z/x)}(log[q(x/z)]) - KL[p(z/x)//q(z)] \cdots (3)
$$
根据定义有：
$$
\begin{split}
& ELBO[p(z/x),q(x/z),q(z)] \\\\
= & \int p(z/x) log[q(x/z)] dz -\int p(z/x)log\frac{p(z/x)}{q(z)}dz \\\\
= & \int p(z/x)log\frac{q(x,z)}{p(z/x)}dz \\\\
= & \int p(z/x)log\frac{q(z/x)q(x)}{p(z/x)}dz \\\\
= & \int p(z/x) log[q(x)] dz -\int p(z/x)log\frac{p(z/x)}{q(z/x)}dz \\\\
= & log[q(x)]\int p(z/x) dz -\int p(z/x)log\frac{p(z/x)}{q(z/x)}dz \\\\
= & log[q(x)] -KL[p(z/x)//q(z/x)] \leq log[q(x)]
\end{split} 
$$

通过采样来计算在$x \sim p(x)$分布下的值

假设$q(z)$服从标准正态分布$N(0,I)$，$p(z/x)$服从各分量独立的正态分布

$$
q(z_{i}) = \frac{1}{\sqrt{2 \pi }}e^{-\frac{1}{2}z_{i}^2} , i\in 1 .. d
$$

$$
p(z_{i}/x)=\frac{1}{\sqrt{2 \pi \sigma_{i}^2(x) }}e^{-\frac{1}{2}(\frac{z_{i}-\mu_{i}(x)}{\sigma_{i}(x)})^2}  , i\in 1 .. d
$$

KL散度：
$$
\begin{split}
KL[p(z_i/x)//q(z_i)] & =\frac 1 2 [\frac{[\sigma_p^2(x)]_i}{1} - ln(\frac{[\sigma_p^2(x)]_i}{1}) + \frac{([\mu_p(x)]_i-0)^2}{1} - 1 ] \\\\
&=\frac 1 2 [[\sigma_p^2(x)]_i - ln([\sigma_p^2(x)]_i) + [\mu_p(x)]^2_i - 1 ]
\end{split}
$$

$$
KL[p(z/x)//q(z)] = \frac 1 2 \sum_{i=1}^d  [[\sigma_p^2(x)]_i - ln([\sigma_p^2(x)]_i) + [\mu_p(x)]^2_i - 1 ]
$$


假设输入的特征向量$x$是连续型变量，$q(x/z)$也是服从各分量独立的正态分布

$$
q(x_{j}/z)=\frac{1}{\sqrt{2 \pi \sigma_{j}^2(z) }}e^{-\frac{1}{2}(\frac{x_{j}-\mu_{j}(z)}{\sigma_{j}(z)})^2}  , j\in 1 .. D
$$

$$
ln[q(x_{j}/z)]=-\frac{1}{2}[ln(2\pi)+ ln(\sigma_{j}^2(z)) + \frac{(x_{j}-\mu_{j}(z))^2}{\sigma_{j}^2(z)} ]
$$

为了方便控制，$\sigma_{j}^2(z)$固定为常量$\sigma^2_g$

$$
\begin{split}
min \\{ -ln[q(x/z)] \\} &= min \\{ \frac{1}{\sigma^2_g} \sum_{j=1}^{D}[x_j-\mu_j(z)]^2 \\} \\\\
&= min \\{ \frac{1}{\sigma^2_g} \big{\lVert} x-\mu(z) \big{\rVert}^2 \\}
\end{split}
$$

p(z/x)和q(x/z)是两个不同的随机变量，不能直接定义交叉熵（必须是同一个随机变量的不同分布），因此$E_{z \sim p(z/x)} log[q(x/z)]$不能写成交叉熵！需要通过采样进行计算:

$$
\begin{split}
min \\{ E_{z \sim p(z/x)} [-ln(q(x/z))] \\} & \approx min \\{ \frac{1}{k} \sum_{j=1}^{k}ln[q(x/z_j)] \\} \\\\
&= min \\{ \frac{1}{k\sigma^2_g} \sum_{j=1}^{k} \big{\lVert} x-\mu(z_j) \big{\rVert}^2 \\} \\\\
& 采样一次，因为在微批中训练可累加采样次数 \\\\
&= min \\{ \frac{1}{\sigma^2_g}  \big{\lVert} x-\mu(z) \big{\rVert}^2 \\} \\\\
&z \sim p(z/x)
\end{split}
$$

两个正态分布的KL散度和交叉熵在下一节中推导具体表达式。



## 相关数学基础

#### 多元积分换元法

多元积分换元法：$x=\phi(y):R^n \to R^n$
$$
\int_D f(x)dx = 
\int_{D'}f(y)|det(\frac{\partial x}{\partial y})|dy 
$$

其中编导数矩阵

$$
\frac{\partial x}{\partial y} = 
\begin{bmatrix}
\frac{\partial x_1}{\partial y_1} & \frac{\partial x_1}{\partial y_2} & \cdots & \frac{\partial x_1}{\partial y_n} \\\\
\frac{\partial x_2}{\partial y_1} & \frac{\partial x_2}{\partial y_2} & \cdots & \frac{\partial x_2}{\partial y_n} \\\\
\vdots&\vdots&\ddots&\vdots \\\\
\frac{\partial x_n}{\partial y_1} & \frac{\partial x_n}{\partial y_2} & \cdots & \frac{\partial x_n}{\partial y_n}
\end{bmatrix}
$$
称为雅可比矩阵

推导如下：
多元变量$x: R^n \to R$全微分定义为

$$
\begin{split}
dx & =\frac{\partial x}{\partial y_1}dy_1+ \frac{\partial x}{\partial y_2}dy_2+\cdots+\frac{\partial x}{\partial y_n}dy_n \\\\
& =(\frac{\partial x}{\partial y_1}, \frac{\partial x}{\partial y_2},\cdots,\frac{\partial x}{\partial y_n})
\begin{pmatrix}
dy_1 \\\\
dy_2 \\\\
\cdots \\\\
dy_n
\end{pmatrix}
\end{split}
$$

于是对于$x=\phi(y), R^n \to R^n$，令$x_i=\phi_i(y)=\pi_i \circ \phi (y)$

$$
\begin{pmatrix}
dx_1 \\\\
dx_2 \\\\
\cdots \\\\
dx_n
\end{pmatrix}=
\begin{bmatrix}
\frac{\partial x_1}{\partial y_1} & \frac{\partial x_1}{\partial y_2} & \cdots & \frac{\partial x_1}{\partial y_n} \\\\
\frac{\partial x_2}{\partial y_1} & \frac{\partial x_2}{\partial y_2} & \cdots & \frac{\partial x_2}{\partial y_n} \\\\
\vdots&\vdots&\ddots&\vdots \\\\
\frac{\partial x_n}{\partial y_1} & \frac{\partial x_n}{\partial y_2} & \cdots & \frac{\partial x_n}{\partial y_n}
\end{bmatrix}
\begin{pmatrix}
dy_1 \\\\
dy_2 \\\\
\cdots \\\\
dy_n
\end{pmatrix}
$$

因此把坐标系转换后两者大小的比例刚好是行列式绝对值代表的容积。

对于二维直角坐标系对应的极坐标系有$x=r\circ cos(\theta),y=r \circ sin(\theta)$，雅可比矩阵为：

$$
\begin{bmatrix}
cos(\theta) & -r \circ sin(\theta) \\\\
sin(\theta) & r \circ cos(\theta)
\end{bmatrix}=r
$$

#### 高斯积分和高斯分布

对二维标准高斯分布做极坐标系转换得到：
$$
\begin{split}
\iint e^{-x^2-y^2}dxdy & = \int_0^{2\pi} \int_0^{+\infty}e^{-r^2}rdrd\theta \\\\
& = \int_0^{2\pi} [-\frac{1}{2} \int_{0}^{-\infty}e^{-r^2}d(-r^2)]d\theta \\\\
& = \int_0^{2\pi}[-\frac{1}{2}e^{-r^2}\Bigg\lvert_{0}^{+\infty}]d\theta \\\\
& = \int_0^{2\pi}\frac{1}{2}d\theta \\\\
& = \pi
\end{split}
$$

另一方面
$$
\begin{split}
\iint e^{-x^2-y^2}dxdy & = [\int e^{-x^2}dx] \circ [\int e^{-y^2}dy] \\\\
& = [\int e^{-x^2}dx] ^2 = \pi
\end{split}
$$

因此高斯积分$\int e^{-x^2}dx=\sqrt{\pi}$


正态分布的概率密度合理化证明：
$\int_{-\infty}^{+\infty} \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2} } dx =1 ?$

令$y=\frac{(x-\mu)}{\sqrt{2}\sigma}$,$dx=\sqrt{2}\sigma dy$

$$
\begin{split}
\int_{-\infty}^{+\infty} \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2} } dx & = \int_{-\infty}^{+\infty} \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-y^2 } \sqrt{2}\sigma dy \\\\
& = \frac{1}{\sqrt\pi} \int_{-\infty}^{+\infty} e^{-y^2}dy = 1
\end{split}
$$


正态分布的熵：
$$
\begin{split}
H(p) & = - \int p(x) ln[p(x)] dx \\\\
& = -\int p(x) ln[\frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2} }] dx \\\\
& = \int \frac{(x-\mu)^2}{2\sigma^2} p(x)dx + \int ln[\sqrt{2 \pi \sigma^2}] p(x) dx \\\\
&= \frac{1}{2\sigma^2} \int (x-\mu)^2 p(x) dx + ln[\sqrt{2 \pi \sigma^2}] \int p(x) dx \\\\
&= \frac{1}{2\sigma^2} E_{x \sim p(x)}[(x-\mu)^2] + ln[\sqrt{2 \pi \sigma^2}] \\\\
&= \frac{1}{2\sigma^2} \sigma^2 + ln[\sqrt{2 \pi \sigma^2}]= ln[\sqrt{2 \pi \sigma^2}] + \frac 1 2
\end{split}
$$

KL散度非负，且当且仅当两个分布相同时为零：

$$
\begin{split}
& KL[p(x)//q(x)]=-E_{x \sim p(x)}[log\frac{q(x)}{p(x)}] \\\\
& log是上凸函数（凹函数），根据Jensen不等式 \\\\
& \geq - log \\{ E_{x \sim p(x)}[\frac{q(x)}{p(x)}] \\} = log \int p(x) \frac{q(x)}{p(x)} dx  \\\\
& = log 1 = 0
\end{split}
$$

正态分布的KL散度：

$$
\begin{split}
& KL[p//q] = -E_{x \sim p(x)}[ln\frac{q(x)}{p(x)}] = -E_{x \sim p(x)}[ln[q(x)]+ln[p(x)]] \\\\&
= -E_{x \sim p(x)}[ln[q(x)]] - H(p)= -\int p(x) ln[q(x)] dx - H(p) \\\\&
= -\int p(x) ln [\frac{1}{\sqrt{2 \pi \sigma_q^2}} e^{- \frac{(x-\mu_q)^2}{2\sigma_q^2}  }] dx -H(p) \\\\&
= \int p(x) ln[\sqrt{2 \pi \sigma_q^2}] dx + \int  \frac{(x-\mu_q)^2}{2\sigma_q^2} p(x) dx  -H(p) \\\\&
= \frac{1}{2 \sigma_q^2} \int [ (x-\mu_p) + (\mu_p-\mu_q) ]^2 p(x) dx + ln(\sqrt{2 \pi \sigma_q^2}) - H(p) \\\\&
= \frac{1}{2 \sigma_q^2} \int [ (x-\mu_p)^2 +2(x-\mu_p)(\mu_p-\mu_q) + (\mu_p-\mu_q)^2 ] p(x) dx + ln(\sqrt{2 \pi \sigma_q^2}) - H(p) \\\\&
= \frac{1}{2 \sigma_q^2} [ \int (x-\mu_p)^2 p(x) dx + \int 2(x-\mu_p)(\mu_p-\mu_q) p(x)dx + \int (\mu_p-\mu_q)^2 p(x)dx] + ln(\sqrt{2 \pi \sigma_q^2}) - H(p) \\\\&
= \frac{1}{2 \sigma_q^2} \\{E_{x \sim p(x)}[(x-\mu_p)^2] + E_{x \sim p(x)}[2(x-\mu_p)(\mu_p-\mu_q)] + E_{x \sim p(x)}[(\mu_p-\mu_q)^2] \\} + ln(\sqrt{2 \pi \sigma_q^2}) - H(p) \\\\&
= \frac{1}{2 \sigma_q^2} \\{\sigma_p^2+ 2(\mu_p-\mu_q)E_{x \sim p(x)}[x-\mu_p] + (\mu_p-\mu_q)^2 \\} + ln(\sqrt{2 \pi \sigma_q^2}) - H(p) \\\\&
= \frac{1}{2 \sigma_q^2} \\{\sigma_p^2+ 2(\mu_p-\mu_q)E_{x \sim p(x)}[x]-2\mu_p(\mu_p-\mu_q) + (\mu_p-\mu_q)^2 \\} + ln(\sqrt{2 \pi \sigma_q^2}) - H(p) \\\\&
= \frac{1}{2 \sigma_q^2} \\{\sigma_p^2+ 2(\mu_p-\mu_q)\mu_p-2\mu_p(\mu_p-\mu_q) + (\mu_p-\mu_q)^2 \\} + ln(\sqrt{2 \pi \sigma_q^2}) - H(p)\frac{1}{2 \sigma_q^2} \\{\sigma_p^2 + (\mu_p-\mu_q)^2 \\} + ln(\sqrt{2 \pi \sigma_q^2}) - (\frac 1 2 + ln(\sqrt{2 \pi \sigma_p^2})) \\\\&
= \frac{\sigma_p^2}{2 \sigma_q^2} + \frac{(\mu_p-\mu_q)^2}{2\sigma_q^2} +  ln(\frac{\sigma_q}{\sigma_p}) - \frac 1 2 \\\\&
= \frac 1 2 [\frac{\sigma_p^2}{\sigma_q^2} - ln(\frac{\sigma_p^2}{\sigma_q^2}) + \frac{(\mu_p-\mu_q)^2}{\sigma_q^2} - 1 ]
\end{split}
$$

正态分布的交叉熵 $CrossEntropy(p//q)$ 记为 $H(p // q)$，根据定义：

$$KL(p//q) = H(p//q) - H(p)$$

因此
$$
\begin{split}
& H(p//q) =KL(p//q)-H(p) = \\\\
&=\frac 1 2 [\frac{\sigma_p^2}{\sigma_q^2} - ln(\frac{\sigma_p^2}{\sigma_q^2}) + \frac{(\mu_p-\mu_q)^2}{\sigma_q^2} - 1 ] - (ln[\sqrt{2 \pi \sigma_p^2}] + \frac 1 2) \\\\
&=\frac 1 2 [\frac{\sigma_p^2}{\sigma_q^2} - ln(\frac{\sigma_p^2}{\sigma_q^2}) + \frac{(\mu_p-\mu_q)^2}{\sigma_q^2} -ln(\sigma_p^2) -ln(2\pi) - 2 ]
\end{split}
$$

#### 幂均值函数及其不等式
幂均值函数定义为，给定$x_i \geq 0$：
$$
M(p) = (\frac {\sum\limits_{k=1}^{n}x_i^p} n)^{\frac 1 p}
$$

幂均值函数与最大最小值
$$
\begin{split}
\lim\limits_{p \rightarrow 0} M(p) & = \prod\limits_{k=1}^{n}x_i^{\frac 1 n} \\\\
\lim\limits_{p \rightarrow \infty} M(p) & = max\\{ x_i \\} \\\\
\lim\limits_{p \rightarrow -\infty} M(p) & = min\\{ x_i \\}
\end{split}
$$
第一个极限使用洛必达法则证明，后面两个极限是对偶的，使用夹逼定理证明：
$$
\begin{split}
&\lim\limits_{p \rightarrow 0} M(p) \\\\
= & \lim\limits_{p \rightarrow 0} (\frac {\sum\limits_{k=1}^{n}x_i^p} n)^{\frac 1 p} \\\\
= & \lim\limits_{p \rightarrow 0} e^{ln(\frac {\sum\limits_{k=1}^{n}x_i^p} n)^{\frac 1 p}} \\\\
= & e^{\lim\limits_{p \rightarrow 0} ln(\frac {\sum\limits_{k=1}^{n}x_i^p} n)^{\frac 1 p}} \\\\
= & e^{\lim\limits_{p \rightarrow 0} {\frac {ln(\frac {\sum\limits_{k=1}^{n}x_i^p} n)} p}} \\\\
= & e^{\lim\limits_{p \rightarrow 0} {\frac {ln(\frac {\sum\limits_{k=1}^{n}x_i^p} n)'} {p'}}} \\\\
= & e^{\lim\limits_{p \rightarrow 0} {\frac {(\frac {\sum\limits_{k=1}^{n}x_i^p} n)^{-1}(\frac {\sum\limits_{k=1}^{n}x_i^p} n)'} {1}}} \\\\
= & e^{\lim\limits_{p \rightarrow 0} {\frac {\sum\limits_{k=1}^{n}ln(x_i)x_i^p} {\sum\limits_{k=1}^{n}x_i^p}}} \\\\
= & e^{ {\frac {\sum\limits_{k=1}^{n}ln(x_i)x_i^0} {\sum\limits_{k=1}^{n}x_i^0}}} \\\\
= & e^{ \frac {\sum\limits_{k=1}^n ln(x_i)} n }=\prod\limits_{k=1}^{n}x_i^{\frac 1 n}
\end{split}
$$
第二个极限，记$X_M=max\\{ x_i \\}$：
$$
\begin{split}
(\frac {X_M^p} n)^{\frac 1 p} \leq (\frac {\sum\limits_{k=1}^{n}x_i^p} n)^{\frac 1 p} \leq (\frac {\sum\limits_{k=1}^{n}X_M^p} n)^{\frac 1 p} \\\\
X_M(\frac 1 n)^{\frac 1 p} \leq M(p) \leq X_M \\\\
\lim\limits_{p \rightarrow \infty} X_M(\frac 1 n)^{\frac 1 p} \leq \lim\limits_{p \rightarrow \infty} M(p) \leq X_M \\\\
\lim\limits_{p \rightarrow \infty} X_M(\frac 1 n)^{\frac 1 p} \leq \lim\limits_{p \rightarrow \infty} M(p) \leq X_M \\\\
X_M \leq \lim\limits_{p \rightarrow \infty} M(p) \leq X_M
\end{split}
$$
第三个极限只要利用$p \rightarrow -\infty <=> q=(-p) \rightarrow \infty $以及$[max\\{ x_i^{-1} \\} ] ^{-1}=min\\{ x_i \\}$

幂均值函数是递增函数，并且有：
$$
\begin{split}
M(-1) = H_n & = (\frac {\sum\limits_{k=1}^{n}x_i^p} n)^{\frac 1 p} \quad 调和平均 \\\\
M(0) = G_n & = (\frac {\sum\limits_{k=1}^{n}x_i^p} n)^{\frac 1 p} \quad 几何平均 \\\\
M(1) = A_n & = (\frac {\sum\limits_{k=1}^{n}x_i^p} n)^{\frac 1 p} \quad 算术平均 \\\\
M(1)= Q_n & = (\frac {\sum\limits_{k=1}^{n}x_i^p} n)^{\frac 1 p} \quad 平方平均 \\\\
min \leq H_n  & \leq G_n\leq A_n \leq Q_n \leq max
\end{split}
$$

TODO 递增证明

#### 柯西不等式和对数不等式
柯西不等式有限元版本：给定两个向量$a,b \in R^n$，则
$$
|a| \cdot |b| \geq |a \cdot b|
$$
这个也是线性代数关于模长的性质：$\frac {|a \cdot b|} {|a| \cdot |b|} = cos(\theta)$，其中$\theta$是两个向量的夹角。

柯西不等式积分版本：给定两个可积函数$f(x)$和$g(x)$：
$$
\int_{a}^{b}|f(x)|^2 dx \cdot \int_{a}^{b}|g(x)|^2 dx \geq |\int_{a}^{b}f(x) g(x) dx |^2
$$

证明对数不等式，给定$b > a > 0$，$(ab)^{\frac 1 2} \leq \frac {b-a} {ln(b) - ln(a)} \leq \frac {a+b} 2$。

有多种证明方法，使用柯西不等式积分版本是最直观的一种。
令$f(x)=\sqrt{x},g(x)=\frac 1 {\sqrt{x}}$，代入柯西不等式有：
$$
\begin{split}
\int_{a}^{b}x dx \cdot \int_{a}^{b} \frac 1 x dx \geq |\int_{a}^{b} dx |^2 \\\\
(b^2/2-a^2/2) \cdot (ln(b)-ln(a)) \geq (b-a)^2 \\\\
\frac {a+b} 2 \geq \frac {b-a} {ln(b) - ln(a)}
\end{split}
$$
令$f(x)=\frac 1 x,g(x)=1$，代入柯西不等式有：
$$
\begin{split}
\int_{a}^{b} {\frac 1 {x^2}} dx \cdot \int_{a}^{b} dx \geq |\int_{a}^{b} \frac 1 x dx |^2 \\\\
(-b^{-1}+a^{-1}) \cdot (b-a) \geq (ln(b)-ln(a))^2 \\\\
\frac {(b-a)^2} {ab} \geq (ln(b)-ln(a))^2 \\\\
\frac {(b-a)^2} {(ln(b)-ln(a))^2} \geq ab \\\\
\frac {b-a} {ln(b)-ln(a)} \geq \sqrt{ab}
\end{split}
$$

#### 最大值函数的平滑逼近
最大值函数$max(x,y)$不是光滑的，固定其中一个变量的值(例如$y$)，则在$x=y$点的导数不存在。

实现最大值函数的平滑逼近的主要思路如下：假设存在函数$f(x)$及其逆函数存在并处处可导，且对任意$k \in Int$有：$f^{-1}(k x)=g(k)+f^{-1}(x)$，其中$g(k)$不是无穷大。则$max \\{ x_i \\} =\lim\limits_{k \rightarrow +\infty}\frac {f^{-1}(\sum\limits_{l=1}^{n} f(k x_i)} k$。证明：
$$
\begin{split}
& \lim\limits_{k \rightarrow +\infty}\frac {f^{-1}(\sum\limits_{l=1}^{n} f(k x_i)} k \\\\
\leq & \lim\limits_{k \rightarrow +\infty}\frac {f^{-1}(f(k m)} k \\\\
= & \lim\limits_{k \rightarrow +\infty}\frac {g(n)+f^{-1}f(km)} k \\\\
= & \lim\limits_{k \rightarrow +\infty}\frac {g(n)} k + m = m \\\\
& \lim\limits_{k \rightarrow +\infty}\frac {f^{-1}(\sum\limits_{l=1}^{n} f(k x_i)} k  \\\\
\geq & \lim\limits_{k \rightarrow +\infty}\frac {f^{-1}(f(k m)} k = m
\end{split}
$$
固定$k$则其导数是：
$$
[\frac {f^{-1}(\sum\limits_{l=1}^{n} f(k x_i)} k]'\approx [\frac {f^{-1}f(km)} k]'= \frac {{f^{-1}}'[f(km)] f'(km)} k
$$

实际使用中指数/对数是常用的平滑选择。令$f(x)=e^x,f^{-1}(x)=ln(x)$，$k=1$，则$max\\{ x_i \\} \approx ln(\sum\limits_{l=1}^{n}e^{x_i})$。当$k$选一些更大的数时(例如10)则逼近效果更佳，然而导数的大小正比于$k$，可能导致数值计算溢出。
