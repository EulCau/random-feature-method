> [线性部分](#线性随机特征法)与[非线性部分](#非线性部分技术路线)跳转超链接

# 线性随机特征法

本项目实现了一种结合 **随机特征方法 (Random Feature Method, RFM)** 与 **Deep BSDE** 框架的算法. 该方法的核心在于利用线性算子结构, 将 PDE 的求解转化为一个高效的线性最小二乘问题. 本项目会生成两个可执行文件, 一个用于实现半线性 Deep BSDE 方法, 另一个用于实现线性随机特征法.

> 线性随机特征法流程草稿见 `./fig`, 其中的不同变量使用不同颜色标注.  

## 1. 算法背景

传统的 Deep BSDE 方法 ([arXiv:1707.02568](https://arxiv.org/abs/1707.02568)) 使用神经网络逼近未知梯度项, 通常需要耗时的梯度下降优化. 本项目通过固定随机特征层的参数, 仅对输出权重进行线性回归, 极大地提升了计算效率.

## 2. 线性 $f$ 的求解过程

### 2.1 采样与初始化

1. **路径采样**: 生成 $n$ 条长度为 $N$ 的随机路径 $X$, 时间步长为 $\Delta t$.

2. **随机特征映射**: 随机采样并固定隐藏层参数 $A, \vec{b}, \vec{c}$ (矩阵和向量行数均为: `hidden_dim`).

3. **待求变量**:
   - $\alpha$: 线性输出权重矩阵 (`dim` $\times$ `hidden_dim`).
   - $y_{0}$: 初始点处的函数值 $u(0, \vec{x_{0}})$.

### 2.2 系统演化

假设已知函数 $f$ 满足线性结构:

$$
f\left(t, \vec{x}, y, \vec{z}\right) = \mathcal{L}\left(t, \vec{x}\right) y + \langle \vec{\mathcal{M}}\left(t, \vec{x}\right), \vec{z} \rangle + \mathcal{N}\left(t, \vec{x}\right)
$$

定义特征向量 $\vec{H_{k}} = \tanh\left(A \vec{x_{k}} + \vec{b} t_{k} + \vec{c}\right)$, 则梯度项 $\langle \sigma\left(t_{k}, x_{k}\right), \nabla u\left(t_{k}, x_{k}\right) \rangle$ 近似为 $\vec{z_{k}} = \alpha \vec{H_{k}}$.

每一步的状态更新公式为:

$$
y_{k+1} = \left(1 - \Delta t \mathcal{L}\left(t_{k}, \vec{x_{k}}\right)\right) y_{k} + \left(\mathrm{d}\vec{w_{k}} - \Delta t \vec{\mathcal{M}}\left(t_{k}, \vec{x_{k}}\right)\right)^{\top} \alpha \vec{H_{k}} + \mathcal{N}\left(t_{k}, \vec{x_{k}}\right) \Delta t
$$

### 2.3 线性求解

由于 $y_{N}$ 是关于 $y_{0}$ 和 $\alpha$ 的线性函数, 我们只需最小化末端状态与边界条件 $g(x_N)$ 的残差:

$$
\text{argmin}_{y_{0}, \alpha} \sum_{i=1}^{n} \left\lVert \tilde{g}\left(x_{N}^{(i)}\right) - g\left(x_{N}^{(i)}\right) \right\rVert^{2}
$$

而这一步可以使用最小二乘法或者岭回归完成.

## 3. 非线性 $f$ 的求解过程

当 $f\left(t, \vec{x}, y, \vec{z}\right)$ 不再具有 2.2 中的线性结构时, $y_{N}$ 不再是 $\left(y_{0},\alpha\right)$ 的线性函数, 末端拟合必须改为非线性最小二乘:

- 参数向量:

$$
\theta = \begin{bmatrix} y_{0} \\ \mathrm{vec}\left(\alpha\right) \end{bmatrix}
$$

- 残差向量 (对每条路径 $i$):

$$
r_{i}\left(\theta\right) = y_{N}^{\left(i\right)}\left(\theta\right) - g\left(x_{N}^{\left(i\right)}\right)
$$

- 目标函数:

$$
F\left(\theta\right) = \frac{1}{2} \left\lVert r\left(\theta\right)\right\rVert_{2}^{2}
$$

并使用 Levenberg–Marquardt (LM) / Gauss–Newton (GN) 作为最终求解器.

### 3.1 迭代过程 (LM / GN)

每一轮外层迭代 (第 $m$ 轮):

1. **前向计算残差**

   使用当前参数 $\theta^{\left(m\right)}$ 进行一次完整的离散演化得到 $y_{N}^{\left(i\right)}\left(\theta^{\left(m\right)}\right)$, 构造 $r\left(\theta^{\left(m\right)}\right)$ 与 $F\left(\theta^{\left(m\right)}\right)$.

2. **构造线性化子问题并求增量**

   记雅可比 $J\left(\theta\right)=\frac{\partial r}{\partial \theta}$。LM/GN 的增量 $\delta^{(m)}$ 由线性系统决定:

   - Gauss–Newton:

    $$
    \left(J^{\top} J\right) \delta^{\left(m\right)} = -J^{\top} r
    $$

   - Levenberg–Marquardt:

    $$
    \left(J^{\top} J + \lambda I\right) \delta^{\left(m\right)} = -J^{\top} r
    $$

     其中 $\lambda > 0$ 为阻尼 (信赖域) 参数.

3. **更新与阻尼调节**

    $$
    \theta^{(m+1)} = \theta^{(m)} + \delta^{(m)}
    $$

   对于 LM, 若新 loss 明显下降, 则减小 $\lambda$; 若不降或上升, 则拒绝该步骤并增大 $\lambda$ (使更新更接近梯度下降, 避免发散).

### 3.2 计算 $J^{\top} r$ 与 $J^{\top} J v$

> 说明: 这里的关键是**线性子问题只需要算子 $v\mapsto J^{\top} J v$ 与向量 $J^{\top} r$**, 因此计算上无需显式形成 $J$ 或 $J^{\top}J$.

这部分计算基于 LibTorch 的 autograd. 核心思想如下:

- 只实现 **残差前向函数** `r = residual(theta)` (它内部调用 `Equation::f/g` 与随机特征 `phi/H` 完成 $y$ 的离散递推).

- 通过 autograd 从 `residual(theta)` 自动得到:
   - $J^{\top} r$ (一阶反向传播)
   - $J^{\top} J v$ 的矩阵-向量乘 (用于 CG / LSMR 等迭代线性求解器)

#### (1) 计算 $J^Tr, 把它当作 loss 的梯度

定义

$$
F\left(\theta\right) = \frac{1}{2} \left\lVert r\left(\theta\right)\right\rVert^{2} \quad \Rightarrow \quad \nabla_{\theta} F\left(\theta\right) = J\left(\theta\right)^{\top} r\left(\theta\right)
$$

程序上只需:

1. `r = residual(theta)` (前向, 返回 shape 为 `[num_sample]` 的残差向量)

2. `loss = 0.5 * (r*r).sum()` (标量)

3. 对 `theta` 做一次 autograd 求导, 得到 `grad_theta`, 这就是 $J^{\top} r$

需注意:
- `Equation::f(t,x,y,z)`, `Equation::g(t_end,x_end)` 参与构图, 因此不能在该阶段 `detach` 或 `NoGradGuard`.

- `theta` 中的 `y0` 与 `alpha` 必须是 `requires_grad=true` 的张量 (实现上将两者打包为一个向量参数).

#### (2) 计算 $J^{\top} J v$, 并用算子形式传递给 CG （不显式构造 $J$)

LM / GN 的线性子问题可以用迭代法求解 (例如 CG), 它只需要一个线性算子:

$$
A\left(v\right) = \left(J^{\top} J + \lambda I\right) v
$$

在程序中 `A` 是一个 matvec 函数, 输入向量 `v` (与 `theta` 同 shape), 输出 `A(v)`.

实现思路:

- 先得到梯度函数 $g\left(\theta\right) = \nabla_{\theta} F\left(\theta\right) = J^{\top} r$, 并用 `create_graph=true` 让它保持计算图;

- 然后对标量 $g\left(\theta\right) \cdot v$ 再做一次 autograd 求导, 得到:

$$
\nabla_{\theta}\left(g\left(\theta\right) \cdot v\right) \approx J^{\top} J v
$$

  在 LM 阻尼的保护下, 它可作为 GN 子问题的有效 matvec.

最终:

$$
A\left(v\right) = \left(J^{\top} J v\right) + \lambda v
$$

## 接口文档

- 代码接口说明见 `api-reference.md`. 该文档由 codex 阅读代码仓库后生成.

---

# 非线性部分技术路线

## 1. 数学建模

你现在考虑一般半线性/非线性的生成元

$$
f(t,x,y,z),
$$

不再要求它具有线性结构

$$
f(t,x,y,z)=L(t,x)y+\langle M(t,x),z\rangle+N(t,x).
$$

这时, 沿离散 BSDE 递推,

$$
y_{k+1} = y_k \cdot f(t_k,x_k,y_k,z_k)\Delta t + z_k^\top \Delta W_k,
$$

其中仍然用随机特征表示

$$
z_k=\alpha H_k,\qquad H_k=\phi(t_k,x_k).
$$

在非线性情形下，$y_N$ 对 $(y_0,\alpha)$ 是非线性的复合函数。

所以最后一步变为:

$$
r(\theta)=y_N(\theta)-g(x_N), \qquad \min_\theta \frac{1}{2}\left\lVert r(\theta)\right\rVert^{2}.
$$

这里

$$
\theta=
\begin{bmatrix}
y_0 \\
\mathrm{vec}(\alpha)
\end{bmatrix}.
$$

这就是标准的非线性最小二乘.

## 2. 求解方法

### 2.1 数学问题

定义残差向量

$$
r(\theta)=
\begin{bmatrix}
r_1(\theta) \\
\vdots \\
r_n(\theta)
\end{bmatrix},
\qquad
r_i(\theta)=y_N^{(i)}(\theta)-g(x_N^{(i)}).
$$

目标函数

$$
F(\theta)=\frac{1}{2}\left\lVert r(\theta)\right\rVert_{2}^{2}.
$$

那么非线性最终要求解的就是

$$
\theta^\star=\arg\min_\theta F(\theta).
$$

### 2.2 数值算法

依然有两种方法:

* **Gauss–Newton (GN)**：最核心的非线性最小二乘方法.
* **Levenberg–Marquardt (LM)**：GN 的阻尼稳定版, 类似岭回归. 最终肯定是使用这种方法的.

线性化残差:

$$
r(\theta+\delta) \approx r(\theta) + J(\theta)\delta,
$$

其中

$$
J(\theta)=\frac{\partial r}{\partial \theta}.
$$

于是子问题是

$$
\min_\delta |J\delta+r|^2.
$$

对应正规方程形式:

$$
J^T J\,\delta = -J^T r.
$$

LM 则改成

$$
(J^T J+\lambda I)\delta = -J^T r.
$$

更新：

$$
\theta \leftarrow \theta+\delta.
$$

### 2.3 工程实现

真正实现时, 关键在于:

1. 如何写出 `residual(theta)`
2. 如何拿到 $J^T r$
3. 如何拿到 $J^T J v$
4. 如何解 LM 子问题
5. 如何保证数值稳定

## 3. 接下来的工作安排

### 阶段 A: 先做一个小规模基准非线性求解器

目标是证明方法的正确性.

固定较小的 `hidden_dim`, 对于每次迭代写出 `residual(theta)` 并显式构造 $J$, 然后使用直接法解 $J\delta \approx -r$, 也就是 `lstsq`

目的是验证非线性递推没写错, `autograd` 没断图, 验证 loss 能下降, 同时给后续高维实现提供参考结果.

### 阶段 B: 做可扩展版

只实现 $J^T r$ 与 $J^T J v$ 的算子, 然后用 CG/LSMR 解

$$
(J^T J+\lambda I)\delta=-J^T r.
$$

## 4. 变量定义

参数向量

$$
\theta=
\begin{bmatrix}
y_0\\
\mathrm{vec}(\alpha)
\end{bmatrix}.
$$

对每条样本路径 $i$, $y_0^{(i)}=y_0$, 然后对 $k=0,\dots,N-1$,

$$
z_k^{(i)}=\alpha H_k^{(i)},
$$

$$
y_{k+1}^{(i)} = y_k^{(i)} \cdot f(t_k,x_k^{(i)},y_k^{(i)},z_k^{(i)})\Delta t + (z_k^{(i)})^\top \Delta W_k^{(i)}.
$$

最后得到

$$
y_N^{(i)}(\theta).
$$

终端残差：

$$
r_i(\theta)=y_N^{(i)}(\theta)-g(x_N^{(i)}).
$$

残差函数 `residual(theta)` 必须是完整可导的, 不能中途 `detach`.

## 5. 计算上怎么求 $J^Tr$ 和 $J^TJv$

这是非线性的真正技术核心.

### 5.1 $J^T r$

定义

$$
F(\theta)=\frac12|r(\theta)|^2.
$$

则

$$
\nabla_\theta F(\theta)=J^Tr.
$$

所以 $J^Tr$ 根本不用单独手推, 直接就是 loss 对参数的梯度.

实现上使用 forward 算出 `r = residual(theta)` 定义 `loss = 0.5 * (r*r).sum()`, 然后使用 `backward` / `autograd` 拿到 `grad_theta`

### 5.2 $J^TJv$

高维下如果不想显式构 $J$, 但 LM/CG 需要算子

$$
A(v)=(J^TJ+\lambda I)v.
$$

则需要

$$
v\mapsto J^TJv.
$$

对当前 LibTorch 环境, 最实际的路线是先用 `autograd` 得到 $g(\theta)=\nabla F(\theta)=J^Tr$, 并保留计算图, 再用标量 $g(\theta)\cdot v$ 对 $\theta$ 求导, 得到的结果是一个 Hessian-vector product 形式, 在残差平方和结构下可作为 LM/GN 的有效 matvec 使用.


### 5.3 更新策略

如果更新后 loss 下降明显, 则可以适当减小 $\lambda$, 如果更新后 loss 不降, 则增大 $\lambda$, 拒绝这一步.

## 6. 初值怎么选

初值非常重要, 可以考虑如下策略: $\alpha=0$, $y_0$ 取终端均值

$$
\alpha^{(0)}=0,
\qquad
y_0^{(0)}=\frac1n\sum_i g(x_N^{(i)}).
$$

这是保守但稳定的选择.
