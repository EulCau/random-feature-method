# RFM 非线性最小二乘接口参考

## Coefficient

- `L()`: (t: Tensor [S,T,1,1], x: Tensor [S,T,1,D]) -> Tensor [S,T,1,1], 返回线性系数 L(t,x), 公开.
- `M()`: (t: Tensor [S,T,1,1], x: Tensor [S,T,1,D]) -> Tensor [S,T,1,D], 返回线性系数 M(t,x), 公开.
- `N()`: (t: Tensor [S,T,1,1], x: Tensor [S,T,1,D]) -> Tensor [S,T,1,1], 返回线性系数 N(t,x), 公开.

## Equation

- `sample()`: (num_sample: int64) -> [Tensor [num_sample,D,T], Tensor [num_sample,D,T+1]], 采样布朗增量与状态路径, 公开.
- `f()`: (t: Tensor [...], x: Tensor [...], y: Tensor [...], z: Tensor [...]) -> Tensor [...], 返回驱动项 f(t,x,y,z), 公开.
- `g()`: (t: Tensor [...], x: Tensor [...]) -> Tensor [...], 返回终端条件 g(t,x), 公开.
- `coef()`: () -> Coefficient, 返回线性系数对象, 公开.
- `dim()`: () -> int64, 返回状态维度, 公开.
- `total_time()`: () -> float, 返回总时间, 公开.
- `num_time_interval()`: () -> int64, 返回时间步数 T, 公开.
- `delta_t()`: () -> float, 返回时间步长 dt, 公开.
- `sqrt_delta_t()`: () -> float, 返回 sqrt(dt), 公开.

- `dim_`: int64, 状态维度缓存, 可继承.
- `total_time_`: float, 终止时间缓存, 可继承.
- `num_time_interval_`: int64, 时间剖分数缓存, 可继承.
- `delta_t_`: float, 时间步长缓存, 可继承.
- `sqrt_delta_t_`: float, 步长平方根缓存, 可继承.
- `linear_`: bool, 线性方程标记, 可继承.
- `coefficient_`: Coefficient, 系数对象缓存, 可继承.

## RandomFeatureFunction (RFF)

- `RandomFeatureFunction()`: (dim: int64, hidden_dim: int64, device: Device, seed: uint64) -> RandomFeatureFunction, 初始化随机特征参数, 公开.
- `resample_params()`: (seed: uint64) -> void, 重采样 A/b/c, 公开.
- `phi()`: (t: Tensor [1或B,T,1,1], x: Tensor [B,T,1,D]) -> Tensor [B,T,H,1], 计算随机特征映射, 公开.
- `dim()`: () -> int64, 返回输入维度 D, 公开.
- `hidden_dim()`: () -> int64, 返回特征维度 H, 公开.
- `seed()`: () -> uint64, 返回当前随机种子, 公开.
- `A()`: () -> Tensor [D,H], 返回参数 A, 公开.
- `b()`: () -> Tensor [1,H], 返回参数 b, 公开.
- `c()`: () -> Tensor [1,H], 返回参数 c, 公开.

- `dim_`: int64, 输入维度缓存, 可继承.
- `hidden_`: int64, 特征维度缓存, 可继承.
- `seed_`: uint64, 随机种子缓存, 可继承.
- `device_`: Device, 设备缓存, 可继承.
- `A_`: Tensor [D,H], 线性投影参数, 可继承, 对应公开函数: `A()`.
- `b_`: Tensor [1,H], 时间项参数, 可继承, 对应公开函数: `b()`.
- `c_`: Tensor [1,H], 偏置参数, 可继承, 对应公开函数: `c()`.

## RFMSolver

- `RFMSolver()`: (config: Config, eq: Equation, device: Device, seed: uint64) -> RFMSolver, 初始化求解器与缓存, 公开.
- `seed()`: () -> uint64, 返回随机种子, 公开.
- `device()`: () -> Device, 返回运行设备, 公开.
- `compute_txw()`: () -> void, 计算并写入 t_/t_end_/dw_/x_/x_end_, 公开.
- `t()`: () -> Tensor [S,T,1,1], 返回 t_, 公开.
- `t_end()`: () -> Tensor [S,1,1,1], 返回 t_end_, 公开.
- `dw()`: () -> Tensor [S,D,T] (可reshape为 Tensor [S,T,D]), 返回 dw_, 公开.
- `x()`: () -> Tensor [S,T,1,D], 返回 x_, 公开.
- `x_end()`: () -> Tensor [S,1,1,D], 返回 x_end_, 公开.
- `compute_L()`: (t: Tensor [S,T,1,1], x: Tensor [S,T,1,D]) -> void, 计算并写入 L_, 公开.
- `L()`: () -> Tensor [S,T,1,1], 返回 L_, 公开.
- `compute_M()`: (t: Tensor [S,T,1,1], x: Tensor [S,T,1,D]) -> void, 计算并写入 M_, 公开.
- `M()`: () -> Tensor [S,T,1,D], 返回 M_, 公开.
- `compute_N()`: (t: Tensor [S,T,1,1], x: Tensor [S,T,1,D]) -> void, 计算并写入 N_, 公开.
- `N()`: () -> Tensor [S,T,1,1], 返回 N_, 公开.
- `compute_H()`: (t: Tensor [S,T,1,1], x: Tensor [S,T,1,D]) -> void, 计算并写入 H_, 公开.
- `H()`: () -> Tensor [S,T,H,1], 返回 H_, 公开.
- `compute_linear_coef()`: () -> [Tensor [S,1+D*H], Tensor [S,1]], 组装线性系统 A/B, 公开.
- `Solve_linear()`: () -> [Tensor [标量], Tensor [D,H], float], 求解并返回 y0/alpha/rmse, 公开.

- `check_tx_shape()`: (t: Tensor [1或S,T,1,1], x: Tensor [S,T,1,D]) -> void, 校验输入形状, dtype, device, 可继承.

- `config_`: Config, 求解配置, 可继承.
- `equation_`: Equation, 方程对象, 可继承.
- `seed_`: uint64, 随机种子, 可继承, 对应公开函数: `seed()`.
- `device_`: Device, 运行设备, 可继承, 对应公开函数: `device()`.
- `rff_`: RandomFeatureFunction, 随机特征对象, 可继承.
- `t_`: Tensor [S,T,1,1], 时间网格, 可继承, 对应公开函数: `t()`.
- `t_end_`: Tensor [S,1,1,1], 终点时间, 可继承, 对应公开函数: `t_end()`.
- `dw_`: Tensor [S,D,T], 布朗增量, 可继承, 对应公开函数: `dw()`.
- `x_`: Tensor [S,T,1,D], 路径状态, 可继承, 对应公开函数: `x()`.
- `x_end_`: Tensor [S,1,1,D], 终端状态, 可继承, 对应公开函数: `x_end()`.
- `L_`: Tensor [S,T,1,1], L 系数缓存, 可继承, 对应公开函数: `L()`.
- `M_`: Tensor [S,T,1,D], M 系数缓存, 可继承, 对应公开函数: `M()`.
- `N_`: Tensor [S,T,1,1], N 系数缓存, 可继承, 对应公开函数: `N()`.
- `H_`: Tensor [S,T,H,1], 随机特征缓存, 可继承, 对应公开函数: `H()`.
