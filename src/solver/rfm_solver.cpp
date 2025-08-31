#include "rfm_solver.h"

// =====================
// 工具：设置随机种子
// =====================
[[maybe_unused]] void set_random_seed(uint64_t seed) {
    torch::manual_seed(seed);
    std::srand(static_cast<unsigned>(seed));
}

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
        float y0 /* 初值，可扩展为张量 */)
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
    Coefficient coefficient = eq.load_coef();
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
    RandomFeatureFunction rff(d, H, /*seed=*/1234);

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
    torch::Tensor alpha = lstsq_result.solution.squeeze(1);        // (H*d)

    // 预测终点
    torch::Tensor y_pred = torch::matmul(M, alpha) + beta; // (K)
    // 末端 L2 误差
    torch::Tensor terminal_err = torch::norm(y_pred - gT) / std::sqrt((double)K);

    SolveResult result{alpha.cpu(), terminal_err.cpu()};
    return result;
}
