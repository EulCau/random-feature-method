#include "rfm_solver.h"
#include "rff.h"
#include "linear_solve_result.h"

RFMSolver::RFMSolver(
    const Config &config, const std::shared_ptr<Equation> &eq, const torch::Device device, const uint64_t seed)
        : config_(config),
          equation_(eq),
          seed_(seed),
          device_(device),
          rff_(RandomFeatureFunction(config.eqn_config.dim, config.net_config.num_hiddens[0], device_, seed_))
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
    const int64_t T = config_.eqn_config.num_time_interval;
    const int64_t S = config_.net_config.valid_size;

    const auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    const auto t_full = torch::linspace(0, total_time, T + 1, opts);

    const auto t_base = t_full.slice(0, 0, T).reshape({1, T, 1, 1});
    t_ = t_base.expand({S, T, 1, 1}).contiguous();

    const auto t_end_base = t_full.slice(0, T, T + 1).reshape({1, 1, 1, 1});
    t_end_ = t_end_base.expand({S, 1, 1, 1}).contiguous();

    const auto [fst, snd] = equation_->sample(S);

    dw_ = fst.to(device_).contiguous();

    const auto x_all = snd.to(device_).permute({0, 2, 1}).contiguous(); // (S, T+1, D)
    x_ = x_all.index({
        at::indexing::Slice(),
        at::indexing::Slice(0, -1),
        at::indexing::Slice()
    }).unsqueeze(2).contiguous(); // (S, T, 1, D)

    x_end_ = x_all.index({
        at::indexing::Slice(),
        at::indexing::Slice(-1, at::indexing::None),
        at::indexing::Slice()
    }).unsqueeze(2).contiguous(); // (S, 1, 1, D)

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

    TORCH_CHECK(
        result.device().type() == device_.type(),
        "result_L must be on ", device_.type(), ", but got ", result.device().type()
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

    TORCH_CHECK(
        result.device().type() == device_.type(),
        "result_M must be on ", device_.type(), ", but got ", result.device().type()
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

    TORCH_CHECK(
        result.device().type() == device_.type(),
        "result_N must be on ", device_.type(), ", but got ", result.device().type()
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

    TORCH_CHECK(
        result.device().type() == device_.type(),
        "result_H must be on ", device_.type(), ", but got ", result.device().type()
    );

    H_ = result;
}

std::pair<const torch::Tensor, const torch::Tensor> RFMSolver::compute_linear_coef() const
{
    const int64_t S = config_.net_config.valid_size;
    const int64_t T = config_.eqn_config.num_time_interval;
    const int64_t D = equation_->dim();
    const int64_t Hdim = config_.net_config.num_hiddens[0];
    const float dt = equation_->delta_t();

    auto device = L_.device();

    // 压缩到紧凑形状
    const auto L = L_.squeeze(-1).squeeze(-1).contiguous();   // (S, T)
    const auto M = M_.squeeze(2).contiguous();                // (S, T, D)
    const auto N = N_.squeeze(-1).squeeze(-1).contiguous();   // (S, T)
    const auto H = H_.squeeze(-1).contiguous();               // (S, T, H)
    const auto dW = dw_.reshape({S, T, D}).contiguous();      // (S, T, D)

    // 线性递推中的三块
    const auto a  = 1.0f - dt * L;      // (S, T)
    const auto xi = dW - dt * M;        // (S, T, D)
    const auto c  = dt * N;             // (S, T)

    // weights[k] = prod_{j=k+1}^{T-1} a_j
    const auto suffix_inclusive = torch::flip(
        torch::cumprod(torch::flip(a, {1}), 1),
        {1}
    ); // (S, T), suffix_inclusive[:, k] = prod_{j=k}^{T-1} a_j

    auto weights = torch::ones_like(a); // (S, T)
    if (T > 1)
    {
        weights.index_put_(
            {torch::indexing::Slice(), torch::indexing::Slice(0, T - 1)},
            suffix_inclusive.index({
                torch::indexing::Slice(),
                torch::indexing::Slice(1, torch::indexing::None)
            })
        );
    }

    // 矩阵第一块: y0 系数
    auto coef_y0 = a.prod(1, true); // (S, 1)

    // 矩阵第二块: alpha 系数
    // weighted_xi: (S, T, D)
    const auto weighted_xi = xi * weights.unsqueeze(-1);

    // coef_alpha[s] = weighted_xi[s]^T @ H[s] -> (D, H)
    auto coef_alpha = torch::bmm(
        weighted_xi.transpose(1, 2).contiguous(), // (S, D, T)
        H                                          // (S, T, H)
    );                                             // (S, D, H)

    coef_alpha = coef_alpha.reshape({S, D * Hdim}); // (S, D*H)

    // 拼接设计矩阵
    const auto A = torch::cat({coef_y0, coef_alpha}, 1).contiguous(); // (S, 1 + D*H)

    // 右端项
    const auto constant_part = (weights * c).sum(1, true); // (S, 1)
    const auto g_XN = equation_->g(t_end_, x_end_).reshape({S, 1}).to(device);

    const auto B = g_XN - constant_part; // (S, 1)

    TORCH_CHECK(
        A.device().type() == device.type() &&
        B.device().type() == device.type(),
        "A, B must be on ", device_.type(), ", but got ", A.device().type(), " & ", B.device().type())


    return {A, B};
}

std::tuple<torch::Tensor, torch::Tensor, float> RFMSolver::Solve_linear() const
{
    const auto [A, B] = compute_linear_coef();

    const auto result = solve_y0_alpha_ridge_dual(
        A, B,
        config_.eqn_config.dim,
        config_.net_config.num_hiddens[0],
        1e-6
    );

    return result;
}

void RFMSolver::check_tx_shape(
    const torch::Tensor& t,
    const torch::Tensor& x
) const
{
    // check t
    TORCH_CHECK(
        t.dim() == 4,
        "t must be a 4D tensor, got dim = ", t.dim()
        );

    TORCH_CHECK(
        t.dtype() == torch::kFloat32,
        "t must be float32, but got ", t.dtype()
        );

    TORCH_CHECK(
        (t.size(0) == x.size(0) || t.size(0) == 1) &&
        t.size(1) == config_.eqn_config.num_time_interval &&
        t.size(2) == 1 &&
        t.size(3) == 1,
        "Invalid shape for t. Expected (",
        x.size(0), " or 1, ",
        config_.eqn_config.num_time_interval,
        ", 1, 1), but got ", t.sizes()
    );

    // check x
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

    TORCH_CHECK(
        x.device().type() == device_.type() &&
        t.device().type() == device_.type(),
        "x, t must be on ", device_.type(), ", but got ", x.device().type(), " & ", t.device().type()
    );
}
