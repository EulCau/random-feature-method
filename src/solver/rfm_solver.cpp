#include "rfm_solver.h"
#include "rff.h"

// TODO: to GPU
RFMSolver::RFMSolver(const Config &config, const std::shared_ptr<Equation> &eq, const uint64_t seed)
        : config_(config),
          equation_(eq),
          seed_(seed),
          rff_(RandomFeatureFunction(config.eqn_config.dim, config.net_config.num_hiddens[0], seed_))
{
    torch::manual_seed(seed_);
    std::srand(static_cast<unsigned>(seed_));

    compute_txw();
    compute_L(t_, x_);
    compute_M(t_, x_);
    compute_N(t_, x_);
    compute_H(t_, x_);
}

void RFMSolver::compute_txw()
{
    const double total_time = config_.eqn_config.total_time;
    int64_t num_time_interval = config_.eqn_config.num_time_interval;
    const torch::Tensor t_full = torch::linspace(
        0, total_time, num_time_interval + 1, torch::TensorOptions()
    );
    t_ = t_full.slice(0, 0, num_time_interval).reshape({1, num_time_interval, 1, 1});
    t_end_ = t_full.slice(0, num_time_interval, num_time_interval + 1).reshape({1, 1, 1, 1});

    const auto [fst, snd] = equation_->sample(config_.net_config.valid_size);
    dw_ = fst;
    const auto x_all = snd.permute({0, 2, 1});
    x_ = x_all.index({
        at::indexing::Slice(),
        at::indexing::Slice(0, -1),
        at::indexing::Slice()
    }).unsqueeze(2).contiguous();
    x_end_ = x_all.index({
        at::indexing::Slice(),
        at::indexing::Slice(-1, at::indexing::None),
        at::indexing::Slice()
    }).unsqueeze(2).contiguous();

    check_tx_shape(t_, x_);
}

// L, M, N are known, these functions are designed to compute results on all the t_k & x_k

void RFMSolver::compute_L(const torch::Tensor &t, const torch::Tensor &x)
{
    check_tx_shape(t, x);

    const auto result = equation_->coef().L(t, x);

    TORCH_CHECK(
        result.dim() == 4 &&
        result.size(0) == config_.net_config.valid_size &&
        result.size(1) == config_.eqn_config.num_time_interval &&
        result.size(2) == 1 &&
        result.size(3) == 1,
        "Invalid shape for L(t, x). Expected (",
        config_.net_config.valid_size, ", ",
        config_.eqn_config.num_time_interval, ", 1, 1), but got ",
        result.sizes()
    );

    L_ = result;
}

void RFMSolver::compute_M(const torch::Tensor& t, const torch::Tensor& x)
{
    check_tx_shape(t, x);

    const torch::Tensor result = equation_->coef().M(t, x);

    TORCH_CHECK(
        result.sizes() == x.sizes(),
        "Invalid shape for M(t, x). Expected ",
        x.sizes(), ", but got ", result.sizes()
    );

    M_ = result;
}

void RFMSolver::compute_N(const torch::Tensor& t, const torch::Tensor& x)
{
    check_tx_shape(t, x);

    const torch::Tensor result = equation_->coef().N(t, x);

    TORCH_CHECK(
        result.dim() == 4 &&
        result.size(0) == config_.net_config.valid_size &&
        result.size(1) == config_.eqn_config.num_time_interval &&
        result.size(2) == 1 &&
        result.size(3) == 1,
        "Invalid shape for N(t, x). Expected (",
        x.size(0), ", ",
        x.size(1), ", 1, 1), but got ",
        result.sizes()
    );

    N_ = result;
}

void RFMSolver::compute_H(const torch::Tensor& t, const torch::Tensor& x)
{
    check_tx_shape(t, x);

    const torch::Tensor result = rff_.phi(t, x);

    TORCH_CHECK(
        result.size(0) == x.size(0) &&
        result.size(1) == x.size(1) &&
        result.size(2) == config_.net_config.num_hiddens[0] &&
        result.size(3) == 1,
        "Invalid shape for H(t, x). Expected ",
        x.sizes(), ", but got ", result.sizes()
    );

    H_ = result;
}

std::pair<torch::Tensor, torch::Tensor> RFMSolver::Solve() const
{
    int64_t S = config_.net_config.valid_size;
    int64_t T = config_.eqn_config.num_time_interval;
    int64_t D = equation_->dim();
    int64_t H_dim = config_.net_config.num_hiddens[0];
    const double dt = equation_->delta_t();
    auto device = L_.device();

    const torch::Tensor dW = dw_.to(device).reshape({S, T, 1, D});

    const auto a_k = 1.0 - L_ * dt;
    const auto b_k = dW - M_ * dt;
    const auto c_k = N_ * dt;

    // 计算累计乘积权重 W_k = \prod_{j=k+1}^{N-1} a_j
    // 我们使用 flip -> cumprod -> flip 的技巧来实现反向累计乘
    const auto a_flipped = torch::flip(a_k.squeeze(-1).squeeze(-1), {1}); // (S, T)
    const auto cum_a_flipped = torch::cumprod(a_flipped, 1);

    // weights 对应于每个时刻 k 的系数。
    // 对于 y_N = ... + W_k * (b_k * alpha * H_k) + ...
    // W_{T-1} 始终为 1, W_{T-2} = a_{T-1}, 依此类推
    auto weights = torch::ones({S, T}, torch::TensorOptions().device(device));
    if (T > 1) {
        weights.index_put_(
            {torch::indexing::Slice(), torch::indexing::Slice(0, -1)},
            torch::flip(cum_a_flipped.index(
                {torch::indexing::Slice(), torch::indexing::Slice(0, -1)}), {1})
        );
    }
    weights = weights.reshape({S, T, 1, 1});

    // 构建线性方程组 A * theta = B

    // (A) y0 的系数: \prod_{j=0}^{N-1} a_j
    torch::Tensor coef_y0 = torch::prod(a_k.squeeze(-1).squeeze(-1), 1).reshape({S, 1});

    // (B) alpha 的系数:
    // 对于每一时刻，我们需要 b_k^T * H_k^T (外积) 得到 (dim, hiddens) 的矩阵
    // b_k 是 (S, T, 1, D) -> 转置为 (S, T, D, 1)
    // H_k 是 (S, T, H, 1) -> 转置为 (S, T, 1, H)
    const auto b_vec = b_k.transpose(2, 3); // (S, T, D, 1)
    const auto h_vec = H_.transpose(2, 3); // (S, T, 1, H)

    // 使用 einsum 或者 bmm 计算外积并乘以权重后求和
    // (S, T, D, 1) * (S, T, 1, H) -> (S, T, D, H)
    const auto alpha_terms = torch::matmul(b_vec, h_vec);
    auto coef_alpha = (weights.unsqueeze(-1) * alpha_terms).sum(1); // (S, D, H)
    coef_alpha = coef_alpha.reshape({S, D * H_dim}); // 展平 alpha 为向量

    // 合并得到大矩阵 A: (S, 1 + D*H)
    const auto A = torch::cat({coef_y0, coef_alpha}, 1);

    // (C) 目标向量 B: g(X_N) - sum(weights * c_k)
    const torch::Tensor g_XN = equation_->g(t_end_, x_end_).to(device); // (S, 1), TODO: 可能需要重写 g 使其支持批量 x
    const torch::Tensor constant_part = (weights * c_k).sum(1).reshape({S, 1});
    const auto B = g_XN - constant_part;

    // 求解线性最小二乘
    // 使用 gelsy (Complete Orthogonal Factorization) 比较稳健
    // 如果存在正则化需求，可以使用 Ridge 回归替代
    const auto solve_result = torch::linalg_lstsq(A, B);
    const torch::Tensor theta = std::get<0>(solve_result); // (1 + D*H, 1)

    // 拆分结果
    torch::Tensor y0 = theta.index({0, 0});
    torch::Tensor alpha = theta.index({
        torch::indexing::Slice(1, torch::indexing::None),
        0
    }).reshape({D, H_dim});

    return {y0, alpha};
}

void RFMSolver::check_tx_shape(
    const torch::Tensor& t,
    const torch::Tensor& x
) const
{
    // ---- check t ----
    TORCH_CHECK(
        t.dim() == 4,
        "t must be a 4D tensor, got dim = ", t.dim()
        );

    TORCH_CHECK(
        t.dtype() == torch::kFloat32,
        "t must be float32, but got ", t.dtype()
        );

    TORCH_CHECK(
        t.size(0) == 1 &&
        t.size(1) == config_.eqn_config.num_time_interval &&
        t.size(2) == 1 &&
        t.size(3) == 1,
        "Invalid shape for t. Expected (1, ",
        config_.eqn_config.num_time_interval,
        ", 1, 1), but got ", t.sizes()
    );

    // ---- check x ----
    TORCH_CHECK(
        x.dim() == 4,
        "x must be a 4D tensor, got dim = ", x.dim()
    );

    TORCH_CHECK(
        x.dtype() == torch::kFloat32,
        "x must be float32, but got", x.dtype()
        );

    TORCH_CHECK(
        x.size(0) == config_.net_config.valid_size &&
        x.size(1) == config_.eqn_config.num_time_interval &&
        x.size(2) == 1 &&
        x.size(3) == config_.eqn_config.dim,
        "Invalid shape for x. Expected (",
        config_.net_config.valid_size, ", ",
        config_.eqn_config.num_time_interval,
        ", 1, ", config_.eqn_config.dim,
        "), but got ", x.sizes()
    );
}
