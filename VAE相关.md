## VAE核心推导过程

$$
x 
\xrightarrow{p(z/x)} z
\xrightarrow{q(x/z)} x
$$
其中：p是编码器，q是解码器（生成器），z是隐藏变量。加上神经网络函数后如下所示：
$$
x \sim p(x) 
\xrightarrow{Encoder} z \sim p(z/x) \quad z \sim q(z)
\xrightarrow{Decoder} x \sim q(x/z)
$$
并且假设$p(z/x)$服从各分量独立的正态分布，$q(z)$服从标准正态分布$N(0,I)$。

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

为了方便控制，$\sigma_{j}^2(z)$固定为常量$\sigma^2_g$，意味着解码器只要输出$\mu_{q(x/z)}(z)$，而不需要输出方差:

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

将以上两项合并起来得到最终的损失函数为：
$$
\begin{split}
 \frac 1 2 \sum_{i=1}^d  [[\sigma_p^2(x)]_i - ln([\sigma_p^2(x)]_i) + [\mu_p(x)]^2_i - 1 ] + \frac{1}{\sigma^2_g}  \big{\lVert} x-\mu(z) \big{\rVert}^2
\end{split}
$$
有趣的是，损失函数刚好分为编码器的损失和解码器的损失两部分
* 第一项从KL散度$KL[p(z/x)//q(z)]$推导而来，并且只跟编码器的参数有关系
* 第二项从$E_{x \sim p(x),z \sim p(z/x)}[- log[q(x/z)]]$而来，并且只跟解码器相关。

实现时
+ 使用$log[\sigma^2(x)]$而不是$log[\sigma(x)]$或者$\sigma(x)$作为网络拟合的结果，因为这时的值域是全体实数，可以避免使用RELU之类求导不够友好的激活函数。
+ $ln([\sigma_p^2(x)]$和$\mu_p(x)$分别是编码器拟合的方差和均值。
+ $\mu(z)$是网络预测（解码器/生成器）的结果，，方差作为超参数不需要预测，并且定义$reg=\frac{1}{\sigma^2_g}>0$。


两个正态分布的KL散度和交叉熵在[数学知识](./数学知识.md)中推导具体表达式。


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
优化目标继续简化为：
$$
\begin{split}
& min \sum_{y} \iint p(x)p(y/z)p(z/x)log\frac{p(x)p(y/z)p(z/x)}{q(x/z)q(z/y)q(y)}dzdx  \\\\
&= min \sum_{y} \iint p(x)p(y/z)p(z/x) (log[p(x)] + log\frac{p(y/z)p(z/x)}{q(x/z)q(z/y)q(y)} ) dzdx  \\\\
&= min ( \sum_{y} \iint p(x)p(y/z)p(z/x) log[p(x)] dzdx + \sum_{y} \iint p(x)p(y/z)p(z/x) log\frac{p(y/z)p(z/x)}{q(x/z)q(z/y)q(y)} dzdx ) \\\\
&= min ( \int p(x)log[p(x)] (\int [\sum_{y} p(y/z)]p(z/x) dz )dx + \sum_{y} \iint p(x)p(y/z)p(z/x) log\frac{p(y/z)p(z/x)}{q(x/z)q(z/y)q(y)} dzdx ) \\\\
&= min ( \int p(x)log[p(x)]dx + \int p(x) \int \sum_{y} p(y/z)p(z/x) log\frac{p(y/z)p(z/x)}{q(x/z)q(z/y)q(y)} dzdx ) \\\\
& \int p(x)log[p(x)]dx是p(x)的信息熵，不包含任何参数，是一个常量，可以去掉 \\\\
&= min \int p(x) ( \int \sum_{y} p(y/z)p(z/x) log\frac{p(z/x)}{q(z/y)} dz + \int \sum_{y} p(y/z)p(z/x) log\frac{p(y/z)}{q(y)} dz - \int \sum_{y} p(y/z)p(z/x) log[q(x/z)] dz) dx \\\\
&= min \int p(x) ( \int \sum_{y} p(y/z)p(z/x) log\frac{p(z/x)}{q(z/y)} dz + \int \sum_{y} p(y/z)p(z/x) log\frac{p(y/z)}{q(y)} dz - \int p(z/x) log[q(x/z)] dz) dx \\\\
&= min E_{x \sim p(x),z \sim p(z/x)} (\sum_{y} p(y/z) log\frac{p(z/x)}{q(z/y)} + \sum_{y} p(y/z) log\frac{p(y/z)}{q(y)} - log[q(x/z)]) \\\\
&= min E_{x \sim p(x),z \sim p(z/x)} (\sum_{y} p(y/z) log\frac{p(z/x)}{q(z/y)} + KL[p(y/z) // q(y)] - log[q(x/z)])
\end{split}
$$

假设$q(z/y)$服从标准正态分布$N(\mu_y,I)$:

$$
q(z_{i}/y) = \frac{1}{\sqrt{2 \pi }}e^{-\frac{1}{2}(z_{i}-\mu_{y,i})^2} , i\in 1 .. d
$$

$p(z/x)$服从各分量独立的正态分布$N(\mu_x,\sigma_x)$

$$
p(z_{i}/x)=\frac{1}{\sqrt{2 \pi \sigma_{i}^2(x) }}e^{-\frac{1}{2}(\frac{z_{i}-\mu_{i}(x)}{\sigma_{i}(x)})^2}  , i\in 1 .. d
$$
因此
$$
\begin{split}
& log\frac{p(z/x)}{q(z/y)} \\\\
= & log\Pi_{z_i} \frac{p(z_i/x)}{q(z_i/y)} \\\\
= & \sum_{z_i} log \frac{p(z_i/x)}{q(z_i/y)} \\\\
= & \sum_{z_i}[ -log\sigma_i(x) + \frac{1}{2}(z_{i}-\mu_{y,i})^2 -\frac{1}{2}(\frac{z_{i}-\mu_{i}(x)}{\sigma_{i}(x)})^2 ] \\\\
& 由于z_i \sim p(z_i/x)是通过重参数z = \epsilon \circ \sigma(x) + \mu(x)计算得到 \\\\
= & \sum_{i=1}^{d}[\frac{1}{2}(z_{i}-\mu_{y,i})^2 -log[\sigma_i(x)] -\frac{1}{2}\epsilon^2 ]
\end{split}
$$


根据上式，损失函数的第一项简化如下(k是分类数目):
$$
\begin{split}
& min\sum_{y} p(y/z) log\frac{p(z/x)}{q(z/y)} \\\\
= & min\sum_{i=1}^{k} p(y_i/z) \sum_{j=1}^{d}[\frac{1}{2}(z_{j}-\mu_{y,i})^2 -log[\sigma_j(x)] ] \\\\
 & 定义Y=(p(y_1/z), \cdots , p(y_k/z))=p(y/z) \\\\
 & N=\frac{1}{2}(z-\mu_y)^{T}(z-\mu_y)-log[\sigma(x)]  \\\\
 & =(\frac{1}{2}(z_{1}-\mu_{y,1})^2 -log[\sigma_1(x)] , \cdots , \frac{1}{2}(z_{d}-\mu_{y,d})^2 -log[\sigma_d(x)]) \\\\
= & min Y  N
\end{split}
$$

目标函数等价于:
$$
\begin{split}
& min E_{x \sim p(x),z \sim p(z/x)} (\sum_{y} p(y/z) log\frac{p(z/x)}{q(z/y)} + KL[p(y/z) // q(y)] - log[q(x/z)]) \\\\
= & min E_{x \sim p(x),z \sim p(z/x)}(Y  N + KL[p(y/z) // q(y)] - log[q(x/z)])
\end{split}
$$

对比标准VAE的损失函数
* 除了$q(z)$变为$q(z/y)$外，原来的$E_{x \sim p(x)}KL[p(z/x)//q(z)]$变成采样$E_{x \sim p(x),z \sim p(z/x)} \sum_{y} p(y/z) log\frac{p(z/x)}{q(z/y)}$，这个变化除了需要$p(z/x)$尽量接近$q(z/y)$外，还要求对每个分类有专属的分布。这部分损失函数属于编码器（主要）和分类器（次要）。
* 而多出来的第二项$E_{x \sim p(x),z \sim p(z/x)} KL[p(y/z) // q(y)]$表示每个分类应该尽量均匀分布而不要重叠坍塌。这部分损失函数属于分类器。
* 最后一项$E_{x \sim p(x),z \sim p(z/x)}[- log[q(x/z)]]$是重构误差，跟标准VAE一致，这部分损失函数属于解码器。

当$q(y)$是均匀分布时，第二项化简如下:

$$
\begin{split}
  & min E_{x \sim p(x),z \sim p(z/x)} KL[p(y/z) // q(y)] \\\\
= & min E_{x \sim p(x),z \sim p(z/x)} \sum_y p(y/z) log \frac{p(y/z)}{q(y)} \\\\
= & min E_{x \sim p(x),z \sim p(z/x)} [\sum_y p(y/z) log[p(y/z)] -\sum_y p(y/z) log [q(y)] \\\\
= & min E_{x \sim p(x),z \sim p(z/x)} [\sum_y p(y/z) log[p(y/z)] -log [q(y)] ] \\\\
= & min E_{x \sim p(x),z \sim p(z/x)} \sum_y p(y/z) log[p(y/z)] 
\end{split}
$$

跟标准VAE一样，$q(x/z)$也是假设服从各分量独立的正态分布
$$
q(x_{j}/z)=\frac{1}{\sqrt{2 \pi \sigma_{j}^2(z) }}e^{-\frac{1}{2}(\frac{x_{j}-\mu_{j}(z)}{\sigma_{j}(z)})^2}  , j\in 1 .. D
$$
为了方便控制，$\sigma_{j}^2(z)$固定为常量$\sigma^2_g$

$$
\begin{split}
min \\{ E_{x \sim p(x),z \sim p(z/x)}[- log[q(x/z)]] \\} & \approx min \\{ \frac{1}{k} \sum_{j=1}^{k}log[q(x/z_j)] \\} \\\\
&= min \\{ \frac{1}{k\sigma^2_g} \sum_{j=1}^{k} \big{\lVert} x-\mu(z_j) \big{\rVert}^2 \\} \\\\
& 采样一次，因为在微批中训练可累加采样次数 \\\\
&= min \\{ \frac{1}{\sigma^2_g}  \big{\lVert} x-\mu(z) \big{\rVert}^2 \\} \\\\
&z \sim p(z/x)
\end{split}
$$

将以上三项合并起来得到最终的损失函数为：
$$
\begin{split}
& p(y/z) N + \sum_y p(y/z) log[p(y/z)] + \frac{1}{\sigma^2_g}  \big{\lVert} x-\mu(z) \big{\rVert}^2 \\\\
 其中: & N=\frac{1}{2}[(z-\mu_y)^{T}(z-\mu_y)-log[\sigma^2(x)]]
\end{split}
$$

+ 实现时$p(y/z)$是分类器预测的向量，$log[p(y/z)]$需要加上一个最小的正常量$keras.epsilon()$避免刚好碰上零值取对数。
+ 使用$log[\sigma^2(x)]$而不是$log[\sigma(x)]$或者$\sigma(x)$作为网络拟合的结果，因为这时的值域是全体实数，可以避免使用RELU之类求导不够友好的激活函数。
+ $\mu(z)$是网络预测（解码器/生成器）的结果，方差作为超参数不需要预测，并且定义$reg=\frac{1}{\sigma^2_g}>0$。
+ 因为$y$有指定的分类数目，因此$\mu_y$定义为一个$k \times d$的矩阵，k是分类数目，d是隐变量的维数。