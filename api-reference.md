# cppRFM/implement/rfm 代码结构与公开接口说明

> 范围：`cppRFM/implement/rfm/src` 下当前已经实现的 **Deep BSDE** 与 **线性 RFM** 相关代码。  
> 目标：说明“有哪些类/公开接口，以及每个类和公开函数做什么”。

## 1. 配置与入口

## 1.1 配置结构体（`config/config.h`）

### `struct EqnConfig`
- `std::string eqn_name`：方程名（用于工厂创建具体方程类）。
- `float total_time`：终止时间 $T$。
- `int64_t dim`：状态维度 $d$。
- `int64_t num_time_interval`：时间剖分步数 $N$。

### `struct NetConfig`
- `std::vector<float> y_init_range`：BSDE 中 `y_init` 随机初始化范围。
- `std::vector<int64_t> num_hiddens`：隐藏层规模配置（MLP/RFF 使用）。
- `std::vector<float> lr_values`、`std::vector<int64_t> lr_boundaries`：学习率相关配置。
- `int64_t num_iterations`：训练步数。
- `int64_t batch_size`、`int64_t valid_size`：训练/验证样本数。
- `std::string dtype`：数据类型字符串配置。
- `bool verbose`：是否打印更多日志。
- `int64_t logging_frequency`：日志打印频率。
- `int64_t warmup_steps`：warmup 步数（学习率调度器使用）。

### `struct Config`
- `EqnConfig eqn_config`
- `NetConfig net_config`

### `Config load_config(const std::string& json_path)`
- 作用：从 JSON 文件读取并构造 `Config`。
- 典型调用：`main_bsde.cpp`、`main_rfm.cpp`。

---

## 1.2 程序入口

### `main_bsde.cpp`
- 调用流程：
  1. `force_link_all_equations()` 注册所有方程；
  2. `load_config(...)` 读取配置；
  3. `EquationFactory::instance().create(...)` 创建方程；
  4. 构造 `BSDESolver` 并 `train()`。

### `main_rfm.cpp`
- 调用流程：
  1. `force_link_all_equations()`；
  2. `load_config(...)`；
  3. 通过工厂创建方程；
  4. 构造 `RFMSolver`，调用 `Solve()` 返回 `(y0, alpha)`。

---

## 2. 方程抽象、工厂与注册机制

## 2.1 系数抽象接口

### `struct Coefficient`（`equations/equation.h`）
用于线性 PDE/RFM 中的系数函数接口：
- `virtual torch::Tensor L(const torch::Tensor& t, const torch::Tensor& x) const = 0;`
- `virtual torch::Tensor M(const torch::Tensor& t, const torch::Tensor& x) const = 0;`
- `virtual torch::Tensor N(const torch::Tensor& t, const torch::Tensor& x) const = 0;`

含义对应线性生成元：
\[
f(t,x,y,z) = L(t,x)y + \langle M(t,x), z \rangle + N(t,x).
\]

## 2.2 方程基类

### `class Equation`（抽象基类）
构造函数：
- `explicit Equation(const EqnConfig& eqn_config)`：初始化 `dim_`、`total_time_`、`delta_t_` 等公共时间网格信息。

公开虚函数：
- `sample(int64_t num_sample)`：采样路径，返回 `(dw, x)`。
- `f(t, x, y, z)`：BSDE 驱动项。
- `g(t, x)`：终端条件。

公开 getter：
- `dim()`、`total_time()`、`num_time_interval()`、`delta_t()`、`sqrt_delta_t()`
- `coef()`：返回线性系数对象（`Coefficient`），供 `RFMSolver` 使用。

## 2.3 方程工厂

### `class EquationFactory`（`equations/equation_factory.h`）
- `using Creator = std::function<std::shared_ptr<Equation>(const EqnConfig&)>;`
- `static EquationFactory& instance()`：单例访问。
- `void register_class(const std::string& name, Creator creator)`：注册方程类型。
- `std::shared_ptr<Equation> create(const std::string& name, const EqnConfig& config) const`：按名称创建方程对象。

### 注册宏与强制链接
- `REGISTER_EQUATION_CLASS(class_name)`：在 `.cpp` 内静态注册方程。
- `force_link_all_equations()`：强制链接 `AllenCahn` / `HJBLQ` / `BSM` 的注册单元，避免链接优化导致注册丢失。

## 2.4 已实现具体方程类（当前均定义在 `.cpp` 内部）

> 这三个类没有单独头文件暴露构造函数；外部通过 `EquationFactory` + 名称字符串创建。

### `AllenCahn`（`allencahn.cpp`）
- `sample(...)`：按加性噪声 SDE 采样路径。
- `f(...) = λ(y - y^3)`。
- `g(...) = 0.5 ||x||^2`。

### `HJBLQ`（`hjblq.cpp`）
- `sample(...)`：按加性噪声 SDE 采样路径。
- `f(...) = -0.5 λ ||z||^2`。
- `g(...) = log((1+||x||^2)/2)`。

### `BSM`（`bsm.cpp`）
- 内部设置 `linear_ = true` 并挂载 `BSMCoefficient`。
- `sample(...)`：几何布朗运动离散采样。
- `f(...)`：当前实现返回 `x * r_`（如需与注释“`-r*y`”一致可后续校对）。
- `g(...)`：`relu(mean(x)-K)`。

### `BSMCoefficient`（`bsm.cpp`）
`Coefficient` 的具体实现：
- `L(...)`：返回常数 `-r` 张量。
- `M(...)`：返回全零（与 `x` 同形状）。
- `N(...)`：返回全零标量场张量。

---

## 3. Deep BSDE 相关模型与求解器

## 3.1 `MLPImpl` / `MLP`（`mlp/mlp.h`）

### 类功能
构造用于逼近 $z_t$ 的子网络：`BatchNorm + Linear + ReLU + ... + Linear + BatchNorm`。

### 公开接口
- `MLPImpl(const Config& config)`：按 `config.net_config.num_hiddens` 构建层。
- `torch::Tensor forward(torch::Tensor x)`：前向传播，输入输出维度都与状态维度 `dim` 对齐。

> `TORCH_MODULE(MLP)` 提供 `MLP` 句柄类型（`std::shared_ptr<MLPImpl>` 风格封装）。

## 3.2 `NonSharedModelImpl` / `NonSharedModel`（`model/non_shared_model.h`）

### 类功能
Deep BSDE 非共享参数模型：
- 可学习初值 `y_init_`、`z_init_`；
- 每个时间步（除最后一步）用一个独立子网络预测 `z`。

### 公开接口
- `NonSharedModelImpl(const Config&, const std::shared_ptr<Equation>&)`：初始化参数与子网络。
- `torch::Tensor forward(const std::pair<torch::Tensor, torch::Tensor>& inputs, bool training)`：执行离散 BSDE 递推并返回末端预测 `y_T`。
- `torch::Tensor y_init() const`：读取当前 `y0` 参数。
- `std::vector<torch::Tensor> parameters_flattened() const`：返回扁平参数列表（用于优化器与梯度裁剪）。

## 3.3 `BSDESolver`（`solver/bsde_solver.h`）

### 类功能
封装 Deep BSDE 的训练流程（采样、损失、反传、优化、学习率调度、日志）。

### 公开接口
- `BSDESolver(const Config&, const std::shared_ptr<Equation>&)`：构造模型、AdamW 优化器与调度器。
- `void train()`：完整训练循环。

### 私有核心方法（理解流程常用）
- `loss_fn(inputs, training)`：Huber 风格截断损失，比较 `model` 输出与 `g(T, X_T)`。
- `train_step(batch)`：一次反向传播 + 梯度裁剪 + 参数更新 + lr 调度。

---

## 4. 线性 RFM 相关类与接口

## 4.1 `RandomFeatureFunction`（`rff/rff.h`）

### 类功能
实现随机特征映射
\[
\phi(x,t)=\tanh(Ax + bt + c)
\]
并固定内层随机参数，仅在外层做线性求解。

### 公开接口
- `RandomFeatureFunction(int64_t dim, int64_t hidden_dim, uint64_t seed=42)`：初始化维度与随机参数。
- `void resample_params(uint64_t seed)`：重采样 `(A,b,c)`。
- `torch::Tensor phi(const torch::Tensor& t, const torch::Tensor& x) const`：计算特征张量。
- getter：`dim()`、`hidden_dim()`、`seed()`、`A()`、`b()`、`c()`。

## 4.2 `RFMSolver`（`solver/rfm_solver.h`）

### 类功能
线性 RFM 主求解器：
1. 采样路径与时间网格；
2. 构造 `L/M/N/H` 张量；
3. 将离散 BSDE 写成线性最小二乘 `A * theta = B`；
4. 求解得到 `(y0, alpha)`。

### 公开接口
- `RFMSolver(const Config& config, const std::shared_ptr<Equation>& eq, uint64_t seed)`：初始化并预计算 `t,x,dw,L,M,N,H`。
- `uint64_t seed() const`：返回随机种子。

- `void compute_txw()`：计算时间网格与样本路径（`t_ / t_end_ / dw_ / x_ / x_end_`）。
- 读取张量：`t()`、`t_end()`、`dw()`、`x()`、`x_end()`。

- `void compute_L(const torch::Tensor& t, const torch::Tensor& x)`，`const torch::Tensor& L() const`。
- `void compute_M(const torch::Tensor& t, const torch::Tensor& x)`，`const torch::Tensor& M() const`。
- `void compute_N(const torch::Tensor& t, const torch::Tensor& x)`，`const torch::Tensor& N() const`。
- `void compute_H(const torch::Tensor& t, const torch::Tensor& x)`，`const torch::Tensor& H() const`。

- `std::pair<torch::Tensor, torch::Tensor> Solve() const`：最小二乘解，返回 `y0` 与 `alpha`。

### `protected` 接口
- `check_tx_shape(const torch::Tensor& t, const torch::Tensor& x) const`：输入形状/类型检查。

---

## 5. 学习率调度工具（BSDE 训练中使用）

## 5.1 `create_lr_lambda(...)`（`solver/lr_scheduler_utils.h`）
- 签名：`std::function<float(int64_t)> create_lr_lambda(int64_t warmup_steps, int64_t total_steps)`
- 功能：返回“线性 warmup + cosine decay”的学习率系数函数。

## 5.2 `LambdaLRScheduler`

### 公开接口
- `LambdaLRScheduler(torch::optim::Optimizer&, std::function<float(int64_t)> lr_lambda)`：绑定优化器与缩放函数。
- `void step()`：推进一步并更新 optimizer 各 param group 的学习率。
- `int64_t step_count() const`：当前步数。

---

## 6. 实际调用关系（简图）

- **Deep BSDE 路径**：
  `main_bsde -> EquationFactory -> BSDESolver -> NonSharedModel(含多个 MLP)`。

- **线性 RFM 路径**：
  `main_rfm -> EquationFactory -> RFMSolver -> Equation::coef + RandomFeatureFunction -> Solve(lstsq)`。

---

## 7. 备注

1. 当前“公开接口”主要来自头文件（`.h`）；`AllenCahn/HJBLQ/BSM` 属于 `.cpp` 内部类，通过工厂间接公开。  
2. `BSM::f` 的代码与注释存在语义差异（注释写 `-r*y`，代码是 `x * r_`），若你准备继续做线性方程实验，建议先确认模型公式再统一。  
3. 若你希望，我可以下一步再给你补一版“**每个接口的输入输出 Tensor 形状总表**”（特别是 `RFMSolver` 这部分，最有用）。
