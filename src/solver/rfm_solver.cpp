#include "rfm_solver.h"

RFMSolver::RFMSolver(const Config &config, const std::shared_ptr<Equation> &eq, const uint64_t seed)
        : config_(config), equation_(eq), seed_(seed)
{
    torch::manual_seed(seed_);
    std::srand(static_cast<unsigned>(seed_));
    auto rff = RandomFeatureFunction(config.eqn_config.dim, config.net_config.num_hiddens[0], seed_);

    L_ = torch::zeros({1, 1}, torch::kFloat32);
    M_ = torch::zeros({1, config.eqn_config.dim}, torch::kFloat32);
    N_ = torch::zeros({1, 1}, torch::kFloat32);
    H_ = torch::zeros({1, config.eqn_config.dim}, torch::kFloat32);
}

// L, M, N are known, these functions are designed to compute results on all the t_k & x_k

void RFMSolver::compute_L(const torch::Tensor &t, const torch::Tensor &x)
{
    torch::Tensor result = torch::zeros({1, 1}, torch::kFloat32);
    // TODO: compute L
    L_ = result;
}

void RFMSolver::compute_M(const torch::Tensor& t, const torch::Tensor& x)
{
    torch::Tensor result = torch::zeros(x.sizes(), torch::kFloat32);
    // TODO: compute M
    M_ = result;
}

void RFMSolver::compute_N(const torch::Tensor& t, const torch::Tensor& x)
{
    torch::Tensor result = torch::zeros({1, 1}, torch::kFloat32);
    // TODO: compute N
    N_ = result;
}

void RFMSolver::compute_H(const torch::Tensor& t, const torch::Tensor& x)
{
    torch::Tensor result = torch::zeros(x.sizes(), torch::kFloat32);
    // TODO: compute H by rff
    H_ = result;
}

/* ai 的胡诌
// =====================
// 构造线性系统 (M, beta)
// M: (K, H*d), beta: (K)
// 递推：
// y_{n+1} = y_n - dt*(a_n y_n + b_n^T z_n + c_n) + <dW_n, z_n>
// z_n = Phi(x_n, t_n)^T alpha  (alpha_flat 拼成 H*d 向量)
// D 更新：D <- D - dt*(a_n D + bDz) + dWDz
// 其中 bDz = [b1*phi, b2*phi, ..., bd*phi], dWDz 类似
// =====================
std::pair<torch::Tensor, torch::Tensor>
build_linear_system_from_paths(
        const Equation& eq,
        RandomFeatureFunction& rff,
        int64_t K,
        float y0)
{
    const int64_t d  = eq.dim();
    const int64_t N  = eq.num_time_interval();
    const float   dt = eq.delta_t();
    const int64_t H  = rff.hidden_dim();

    // 采样路径
    auto [dw, x] = eq.sample(K); // dw:(K,d,N), x:(K,d,N+1)  —— 注意你提供的sample是 (num_sample, dim, num_time_interval) 排列
    // 为了便于索引，我们将用 contiguous 并明确索引顺序
    dw = dw.contiguous();
    x  = x.contiguous();

    // 载入/构造 a,b,c
    Coefficient coefficient = eq.load_coef();  // TODO: linear equation
    auto a = coefficient.a.contiguous();       // (K,N)
    auto b = coefficient.b.contiguous();       // (K,N,d)
    auto c = coefficient.c.contiguous();       // (K,N)

    // 输出：M (K, H*d), beta (K)
    torch::Tensor M     = torch::zeros({K, H * d}, torch::kFloat32);
    torch::Tensor beta  = torch::zeros({K},        torch::kFloat32);

    for (int64_t k = 0; k < K; ++k)
    {
        // beta_k 与 D_k 的初始化
        float beta_k = y0;
        torch::Tensor D_k = torch::zeros({H * d}, torch::kFloat32); // (H*d)

        for (int64_t n = 0; n < N; ++n)
        {
            float t_n = static_cast<float>(n) * dt;

            // 取 x_n^k : 形状 (d)
            torch::Tensor x_kn = x.index({k, torch::indexing::Slice(), n}).clone(); // (d)

            // 计算 phi(x_n, t_n) : (H)
            torch::Tensor phi = rff.phi(x_kn, t_n); // (H)

            // 取 a,b,c, dW
            auto a_kn = a.index({k, n}).item<float>();
            auto c_kn = c.index({k, n}).item<float>();
            torch::Tensor b_kn  = b.index({k, n});            // (d)
            torch::Tensor dW_kn = dw.index({k, torch::indexing::Slice(), n}); // (d)

            // 计算 bDz 与 dWDz （避免构造庞大 Dz，直接用块拼接思想）
            // bDz = [b1*phi, b2*phi, ..., bd*phi]  -> (H*d)
            // dWDz = [dW1*phi, ..., dWd*phi]       -> (H*d)
            torch::Tensor bDz  = torch::empty({H * d}, torch::kFloat32);
            torch::Tensor dWDz = torch::empty({H * d}, torch::kFloat32);
            for (int64_t j = 0; j < d; ++j)
            {
                auto bj  = b_kn.index({j}).item<float>();
                auto dWj = dW_kn.index({j}).item<float>();
                // 拷贝到各自的块
                bDz.index_put_({ torch::indexing::Slice(j*H, (j+1)*H) }, bj  * phi);
                dWDz.index_put_({ torch::indexing::Slice(j*H, (j+1)*H) }, dWj * phi);
            }

            // 递推更新：
            // 常数项 beta:  beta <- beta - dt*(a*beta + c)
            beta_k = beta_k - dt * (a_kn * beta_k + c_kn);

            // 线性系数 D:  D <- D - dt*(a*D + bDz) + dWDz
            D_k = D_k - dt * (a_kn * D_k + bDz) + dWDz;
        }

        // 写入输出
        M.index_put_({k, torch::indexing::Slice()}, D_k);
        beta.index_put_({k}, beta_k);
    }

    return {M, beta};
}

SolveResult solve(const Config& config, const std::shared_ptr<Equation>& eq)
{
    int64_t K = config.net_config.batch_size;   // 用 batch_size 作为采样路径数
    int64_t H = config.net_config.num_hiddens[0]; // 假设 num_hiddens[0] 就是 hidden_dim
    int64_t d = eq->dim();
    int64_t N = eq->num_time_interval();

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    // 随机特征
    RandomFeatureFunction rff(d, H, 1234);

    // 初始化 y0
    torch::Tensor y_init = torch::empty({1})
            .uniform_(config.net_config.y_init_range[0],
                      config.net_config.y_init_range[1]);

    auto y0 = y_init.item<float>();

    // 构造线性系统
    auto [M_cpu, beta_cpu] = build_linear_system_from_paths(*eq, rff, K, y0);
    torch::Tensor M    = M_cpu.to(device);    // (K, H*d)
    torch::Tensor beta = beta_cpu.to(device); // (K)

    // 终端目标 g(X_T)
    auto [dw, x] = eq->sample(K);             // x:(K,d,N+1)
    dw = dw.to(device);
    x  = x.to(device);

    torch::Tensor xT = x.index({torch::indexing::Slice(), torch::indexing::Slice(), N}); // (K,d)
    torch::Tensor tT = torch::full({K}, eq->total_time(), torch::TensorOptions().device(device).dtype(torch::kFloat32));
    torch::Tensor gT = eq->g(tT, xT).to(device); // (K)

    // 最小二乘解: 直接调用 torch::linalg_lstsq
    // A = M, b = gT - beta
    torch::Tensor rhs = gT - beta; // (K)
    auto lstsq_result = torch::linalg_lstsq(M, rhs.unsqueeze(1)); // b 必须是 (K,1)
    torch::Tensor alpha = std::get<0>(lstsq_result).squeeze(1);        // (H*d)

    // 预测终点
    torch::Tensor y_pred = torch::matmul(M, alpha) + beta; // (K)
    // 末端 L2 误差
    torch::Tensor terminal_err = torch::norm(y_pred - gT) / std::sqrt((double)K);

    SolveResult result{alpha.cpu(), terminal_err.cpu()};
    return result;
}


// TODO: check
torch::Tensor compute_A_and_b_for_least_squares(
    const std::vector<std::pair<torch::Tensor, torch::Tensor>>& samples,  // 样本路径
    const RandomFeatureFunction& rf_func,                               // 随机特征
    const Equation& eqn,                                                // 方程
    float delta_t,                                                      // 时间步长
    int64_t N                                                           // 总时间步数
) {
    int64_t K = samples.size();  // 样本数
    int64_t m = rf_func.hidden_dim();  // 特征维度
    int64_t dim = eqn.dim();    // 状态空间维度

    // 初始化矩阵 A 和向量 b
    torch::Tensor A = torch::zeros({K * N, m}, torch::kFloat);  // (K * N) x m
    torch::Tensor b = torch::zeros({K * N}, torch::kFloat);     // (K * N)

    int64_t row_offset = 0;  // 行偏移量

    // 遍历每个样本路径
    for (int64_t k = 0; k < K; ++k) {
        auto [x, w] = samples[k];  // x 是样本路径，w 是相应的随机过程增量
        torch::Tensor y = torch::zeros({x.size(1)});  // 初始化 y_n^k (与路径相同维度)
        torch::Tensor z = torch::zeros_like(y);      // 初始化 z_n^k (与 y 一样)

        // 递推过程：从 n = 0 到 n = N-1
        for (int64_t n = 0; n < N; ++n) {
            float t_n = n * delta_t;  // 计算当前时间点
            torch::Tensor x_n = x.select(1, n);  // 获取路径在当前时间点的状态 x_n^k

            // 计算特征值 z_n^k = φ(x_n^k, t_n)
            torch::Tensor z_n = rf_func.phi(x_n, t_n);

            // 计算 f(t_n, x_n^k, y_n^k, z_n^k)
            torch::Tensor f_vals = eqn.f(t_n, x_n, y, z_n);

            // 更新 y_{n+1}^k: 递推步骤
            y = y - delta_t * f_vals + (w.select(1, n).dot(z_n)); // 用随机过程增量和内积更新

            // 填充矩阵 A 和向量 b
            A.slice(0, row_offset, row_offset + 1) = z_n.view({1, -1});  // 将 φ(x_n^k, t_n) 填充到 A 中
            b.slice(0, row_offset, row_offset + 1) = f_vals.view({1});    // 对应的目标 b 填充

            row_offset += 1;  // 更新行偏移量
        }
    }

    return std::make_pair(A, b);  // 返回大矩阵 A 和右端向量 b
}

torch::Tensor solve_least_squares(const torch::Tensor& A, const torch::Tensor& b) {
    // 使用正规方程来求解：theta = (A^T A)^-1 A^T b
    torch::Tensor AtA = A.transpose(0, 1).matmul(A);  // 计算 A^T * A
    torch::Tensor Atb = A.transpose(0, 1).matmul(b);  // 计算 A^T * b
    torch::Tensor theta = torch::linalg::solve(AtA, Atb);  // 解正规方程

    return theta;  // 返回系数向量
}
*/