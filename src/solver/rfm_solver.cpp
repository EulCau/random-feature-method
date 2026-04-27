#include "rfm_solver.h"

#include <algorithm>
#include <iostream>
#include <utility>
#include "rff.h"
#include "linear_solve_result.h"

RFMSolver::RFMSolver(
    Config config, const std::shared_ptr<Equation> &eq,
    const torch::Device device, const uint64_t seed, const bool is_linear)
        : is_linear_(is_linear),
          config_(std::move(config)),
          equation_(eq),
          seed_(seed),
          device_(device),
          rff_(RandomFeatureFunction(
                config_.eqn_config.dim,
                config_.net_config.num_hiddens[0],
                device_,
                seed_))
{
    torch::manual_seed(seed_);
    std::srand(static_cast<unsigned>(seed_));

    compute_txw();

    if (is_linear_)
    {
        compute_L(t_, x_);
        compute_M(t_, x_);
        compute_N(t_, x_);
    }

    compute_H(t_, x_);
}

/* Options
 * set the initial $y_0$, $alpha$, and $lambda$. */

RFMSolver& RFMSolver::options(
    const std::optional<torch::Tensor>& y0,
    const std::optional<torch::Tensor>& alpha,
    const std::optional<float> lambda
)
{
    const auto D = equation_->dim();
    const auto H = rff_.hidden_dim();

    y0_ = torch::randn({1}, torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(device_));

    alpha_ = torch::randn({D, H}, torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(device_)) * 0.01;

    if (y0.has_value())
    {
        y0_ = y0.value().to(device_).clone().detach();
    }

    if (alpha.has_value())
    {
        alpha_ = alpha.value().to(device_).clone().detach();
    }

    if (lambda.has_value())
    {
        TORCH_CHECK(lambda.value() > 0.0, "lambda must be positive");
        lambda_ = lambda.value();
    }

    return *this;
}

/* Solver
 * `Solver` directs linear and nonlinear problems to different main solver functions. */

std::tuple<torch::Tensor, torch::Tensor, float> RFMSolver::solve(const bool output_log) const
{
    if (is_linear_) return solve_linear();
    return solve_nonlinear(output_log);
}

std::tuple<torch::Tensor, torch::Tensor, float> RFMSolver::solve_linear() const
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

std::tuple<torch::Tensor, torch::Tensor, float> RFMSolver::solve_nonlinear(const bool output_log) const
{
    TORCH_CHECK(y0_.defined(), "y0_ is not initialized");
    TORCH_CHECK(alpha_.defined(), "alpha_ is not initialized");
    TORCH_CHECK(lambda_ > 0.0, "lambda_ must be positive");

    return solve_nonlinear_levenberg_marquardt(y0_, alpha_, lambda_, output_log);
}

/* Utils
 * including calculating intermediate quantities, building the solver, checking tensor status, etc. */

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

std::tuple<torch::Tensor, torch::Tensor, float> RFMSolver::solve_nonlinear_levenberg_marquardt(
    const torch::Tensor &y0, const torch::Tensor &alpha, const float lambda, const bool output_log) const
{
    const int64_t max_iters = config_.net_config.num_iterations;

    torch::Tensor theta = pack_nonlinear_parameters(y0, alpha).detach().clone().to(device_);
    float damping = lambda;
    float final_error = 0.0f;

    for (int64_t iter = 0; iter < max_iters; ++iter)
    {
        constexpr NonlinearSolveOptions options;

        auto theta_with_grad = theta.detach().clone().requires_grad_(true);

        const auto residual = compute_nonlinear_terminal_residual(theta_with_grad).reshape({-1});
        const float curr_loss = 0.5f * residual.pow(2).sum().item<float>();
        const float curr_error = std::sqrt(residual.pow(2).mean().item<float>());

        const auto jacobian = compute_nonlinear_jacobian(residual, theta_with_grad);
        const auto delta = solve_lm_step(jacobian, residual, damping);
        const float step_norm = delta.norm().item<float>();

        const auto trial_theta = (theta + delta).detach();
        const auto trial_residual = compute_nonlinear_terminal_residual(trial_theta).reshape({-1});
        const float trial_loss = 0.5f * trial_residual.pow(2).sum().item<float>();
        const float trial_error = std::sqrt(trial_residual.pow(2).mean().item<float>());
        const bool accepted = trial_loss < curr_loss;

        if (output_log)
        {
            std::cout
                << "[LM] iter=" << iter
                << " loss=" << curr_loss
                << " error=" << curr_error
                << " trial_error=" << trial_error
                << " lambda=" << damping
                << " step_norm=" << step_norm
                << " accepted=" << std::boolalpha << accepted
                << std::noboolalpha
                << std::endl;
        }

        if (accepted)
        {
            theta = trial_theta;
            damping = std::max(options.min_lambda, damping * options.lambda_decrease);
            final_error = trial_error;

            if (final_error <= options.error_tol || step_norm <= options.step_tol)
            {
                break;
            }
        }
        else
        {
            damping = std::min(options.max_lambda, damping * options.lambda_increase);
        }
    }

    const int64_t D = equation_->dim();
    const int64_t Hdim = rff_.hidden_dim();

    const auto final_y0 = theta.index({0}).reshape({1});
    const auto final_alpha = theta.index({
        torch::indexing::Slice(1, torch::indexing::None)
    }).reshape({D, Hdim}).contiguous();
    const auto final_residual = compute_nonlinear_terminal_residual(theta).reshape({-1});
    final_error = std::sqrt(final_residual.pow(2).mean().item<float>());

    return {
        final_y0.detach().clone(),
        final_alpha.detach().clone(),
        final_error
    };
}

torch::Tensor RFMSolver::compute_nonlinear_terminal_residual(
    const torch::Tensor& theta
) const
{
    const int64_t S = config_.net_config.valid_size;
    const int64_t D = equation_->dim();
    const int64_t Hdim = rff_.hidden_dim();
    const int64_t expected_size = 1 + D * Hdim;

    TORCH_CHECK(
        theta.dim() == 1 && theta.size(0) == expected_size,
        "theta must have shape (", expected_size, "), but got ", theta.sizes()
    );

    const auto y0 = theta.index({0}).reshape({1});
    const auto alpha = theta.index({
        torch::indexing::Slice(1, torch::indexing::None)
    }).reshape({D, Hdim}).contiguous();

    const auto y_terminal = forward_nonlinear_terminal_y(y0, alpha);
    const auto x_end_eq = x_end_.permute({0, 3, 1, 2}).contiguous();
    const auto g_terminal = equation_->g(t_end_, x_end_eq);

    TORCH_CHECK(
        g_terminal.sizes() == y_terminal.sizes(),
        "equation_->g must return shape ", y_terminal.sizes(), ", but got ", g_terminal.sizes()
    );

    const auto residual = y_terminal - g_terminal;
    return residual.reshape({S, 1}).contiguous();
}

torch::Tensor RFMSolver::compute_nonlinear_jacobian(
    const torch::Tensor& residual,
    const torch::Tensor& theta
)
{
    const int64_t num_residual = residual.numel();
    const int64_t num_param = theta.numel();
    auto jacobian = torch::zeros(
        {num_residual, num_param},
        theta.options().dtype(theta.dtype())
    );

    for (int64_t i = 0; i < num_residual; ++i)
    {
        auto grad_output = torch::zeros_like(residual);
        grad_output.index_put_({i}, 1.0f);

        auto grads = torch::autograd::grad(
            {residual},
            {theta},
            {grad_output},
            true,
            false,
            false
        );

        jacobian.index_put_({i}, grads[0].reshape({num_param}));
    }

    return jacobian.contiguous();
}

torch::Tensor RFMSolver::solve_lm_step(
    const torch::Tensor& jacobian,
    const torch::Tensor& residual,
    const float lambda
)
{
    TORCH_CHECK(lambda > 0.0f, "lambda must be positive");

    const auto j_t = jacobian.transpose(0, 1).contiguous();
    const auto system = torch::matmul(j_t, jacobian);
    const auto rhs = -torch::matmul(j_t, residual.reshape({-1, 1}));
    const auto identity = torch::eye(
        system.size(0),
        torch::TensorOptions().dtype(system.dtype()).device(system.device())
    );

    return torch::linalg_solve(system + lambda * identity, rhs).reshape({-1}).contiguous();
}

torch::Tensor RFMSolver::forward_nonlinear_terminal_y(
    const torch::Tensor& y0,
    const torch::Tensor& alpha
) const
{
    using namespace torch::indexing;

    const int64_t S = config_.net_config.valid_size;
    const int64_t T = config_.eqn_config.num_time_interval;
    const float dt = equation_->delta_t();

    auto y = y0.reshape({1, 1, 1, 1}).expand({S, 1, 1, 1});
    const auto z_all = compute_nonlinear_z(alpha);
    const auto dw_all = dw_.permute({0, 2, 1}).unsqueeze(2).contiguous(); // (S, T, 1, D)

    for (int64_t k = 0; k < T; ++k)
    {
        const auto t_k = t_.index({
            Slice(), Slice(k, k + 1), Slice(), Slice()});
        const auto x_k = x_.index({
            Slice(), Slice(k, k + 1), Slice(), Slice()});
        const auto z_k = z_all.index({
            Slice(), Slice(k, k + 1), Slice(), Slice()});
        const auto dw_k = dw_all.index({
            Slice(), Slice(k, k + 1), Slice(), Slice()});

        const auto x_eq = x_k.permute({0, 3, 1, 2}).contiguous();
        const auto z_eq = z_k.permute({0, 3, 1, 2}).contiguous();
        const auto f_k = equation_->f(t_k, x_eq, y, z_eq);
        const auto martingale = torch::sum(dw_k * z_k, -1, true);

        TORCH_CHECK(
            f_k.sizes() == y.sizes(),
            "equation_->f must return shape ", y.sizes(), ", but got ", f_k.sizes()
        );

        y = y - dt * f_k + martingale;
    }

    return y.contiguous();
}

torch::Tensor RFMSolver::compute_nonlinear_z(const torch::Tensor& alpha) const
{
    const auto features = H_.squeeze(-1).contiguous(); // (S, T, H)
    return torch::matmul(features, alpha.transpose(0, 1)).unsqueeze(2).contiguous(); // (S, T, 1, D)
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

torch::Tensor RFMSolver::pack_nonlinear_parameters(
    const torch::Tensor& y0,
    const torch::Tensor& alpha
)
{
    return torch::cat({
        y0.reshape({1}),
        alpha.reshape({-1})
    }).contiguous();
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
