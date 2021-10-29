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

两个正态分布的KL散度和交叉熵在[数学知识](./数学知识.md)中推导具体表达式。
