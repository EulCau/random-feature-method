# 线性随机特征法

本项目实现了一种结合 **随机特征方法 (Random Feature Method, RFM)** 与 **Deep BSDE** 框架的算法. 该方法的核心在于利用线性算子结构, 将 PDE 的求解转化为一个高效的线性最小二乘问题. 本项目会生成两个可执行文件, 一个用于实现半线性 Deep BSDE 方法, 另一个用于实现线性随机特征法.

> 线性随机特征法流程草稿见 `./fig`, 其中的不同变量使用不同颜色标注.
>
> 半线性的随机特征法在最终实现的计划中.

## 1. 算法背景

传统的 Deep BSDE 方法 (arXiv:1707.02568) 使用神经网络逼近未知梯度项, 通常需要耗时的梯度下降优化. 本项目通过固定随机特征层的参数, 仅对输出权重进行线性回归, 极大地提升了计算效率.

## 2. 核心数学流程

### 2.1 采样与初始化

1. **路径采样**: 生成 $n$ 条长度为 $N$ 的随机路径 $X$, 时间步长为 $\Delta t$.
2. **随机特征映射**: 随机采样并固定隐藏层参数 $A, \vec{b}, \vec{c}$ (矩阵和向量行数均为: `hidden_dim`).
3. **待求变量**:
   - $\alpha$: 线性输出权重矩阵 (`dim` $\times$ `hidden_dim`).
   - $y_{0}$: 初始点处的函数值 $u(0, \vec{x_{0}})$.

### 2.2 系统演化 (Euler-Maruyama)

假设已知函数 $f$ 满足线性结构:

$$f\left(t, \vec{x}, y, \vec{z}\right) = \mathcal{L}\left(t, \vec{x}\right) y + \langle \vec{\mathcal{M}}\left(t, \vec{x}\right), \vec{z} \rangle + \mathcal{N}\left(t, \vec{x}\right)$$

定义特征向量 $\vec{H_{k}} = \tanh\left(A \vec{x_{k}} + \vec{b} t_{k} + \vec{c}\right)$, 则梯度项 $\langle \sigma\left(t_{k}, x_{k}\right), \nabla u\left(t_{k}, x_{k}\right) \rangle$ 近似为 $\vec{z_{k}} = \alpha \vec{H_{k}}$.

每一步的状态更新公式为:

$$y_{k+1} = \left(1 - \Delta t \mathcal{L}\left(t_{k}, \vec{x_{k}}\right)\right) y_{k} + \left(\mathrm{d}\vec{w_{k}} - \Delta t \vec{\mathcal{M}}\left(t_{k}, \vec{x_{k}}\right)\right)^{\top} \alpha \vec{H_{k}} + \mathcal{N}\left(t_{k}, \vec{x_{k}}\right) \Delta t$$

### 2.3 线性求解

由于 $y_{N}$ 是关于 $y_{0}$ 和 $\alpha$ 的线性函数, 我们只需最小化末端状态与边界条件 $g(x_N)$ 的残差:

$$\text{argmin}_{y_{0}, \alpha} \sum_{i=1}^{n} \left\lVert \tilde{g}\left(x_{N}^{(i)}\right) - g\left(x_{N}^{(i)}\right) \right\rVert^{2}$$

而这步可以使用最小二乘法或者 **Ridge Regression (岭回归)** 完成.
