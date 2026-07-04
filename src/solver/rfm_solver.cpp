#include "rfm_solver.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <tuple>
#include <utility>
#include <vector>
#include "linear_least_squares_solver.h"
#include "nonlinear_solver_utils.h"
#include "rff.h"

namespace
{

[[nodiscard]] std::tuple<torch::Tensor, torch::Tensor, float> split_linear_solution(
    const torch::Tensor& x,
    const int64_t dim,
    const int64_t hidden_dim,
    const float rmse)
{
    const auto x_matrix = x.reshape({-1, 1}).contiguous();
    const auto y0 = x_matrix.index({0, 0}).clone();
    const auto alpha = x_matrix.index({
        torch::indexing::Slice(1, torch::indexing::None),
        0
    }).reshape({dim, hidden_dim}).contiguous();

    return {y0, alpha, rmse};
}

[[nodiscard]] std::pair<torch::Tensor, torch::Tensor> reduce_linear_qr_libtorch(
    const torch::Tensor& A,
    const torch::Tensor& B)
{
    TORCH_CHECK(A.dim() == 2, "A must be 2D, got ", A.sizes());
    TORCH_CHECK(B.dim() == 2, "B must be 2D, got ", B.sizes());
    TORCH_CHECK(A.size(0) == B.size(0), "A and B row mismatch: A=", A.sizes(), ", B=", B.sizes());
    TORCH_CHECK(A.size(0) >= A.size(1), "QR reduction requires rows >= cols, got ", A.sizes());

    const auto n = A.size(1);
    const auto [qr_data, tau] = torch::geqrf(A.contiguous());
    const auto R = qr_data.index({
        torch::indexing::Slice(0, n),
        torch::indexing::Slice()
    }).triu().contiguous();
    const auto transformed_rhs = torch::ormqr(qr_data, tau, B.contiguous(), true, true);
    const auto rhs = transformed_rhs.index({
        torch::indexing::Slice(0, n),
        torch::indexing::Slice()
    }).contiguous();
    return {R.contiguous(), rhs.contiguous()};
}

} // namespace

RFMSolver::RFMSolver(
    const Config& config, const std::shared_ptr<Equation> &eq,
    const torch::Device device, const uint64_t seed)
        : RFMSolver(config, eq, device, seed, config.solver_config.use_linear_solver)
{
}

RFMSolver::RFMSolver(
    Config config, const std::shared_ptr<Equation> &eq,
    const torch::Device device, const uint64_t seed, const bool is_linear)
        : is_linear_(is_linear),
          config_(std::move(config)),
          equation_(eq),
          seed_(seed),
          device_(device),
          rff_(RandomFeatureFunction(
                config_.eqn_config.dimension,
                config_.solver_config.hidden_dim,
                device_,
                seed_)),
          lambda_(config_.solver_config.initial_lambda)
{
    TORCH_CHECK(equation_ != nullptr, "equation must not be null");
    TORCH_CHECK(
        !is_linear || equation_->is_linear(),
        "Linear solver was requested, but equation '",
        config_.eqn_config.equation_name,
        "' is not marked as linear"
    );
    TORCH_CHECK(
        !equation_->is_linear() || equation_->has_coefficient(),
        "Equation '",
        config_.eqn_config.equation_name,
        "' is marked as linear, but L/M/N coefficients are not defined"
    );
    is_linear_ = is_linear && equation_->is_linear();

    torch::manual_seed(seed_);
    std::srand(static_cast<unsigned>(seed_));

    if (is_linear_)
    {
        compute_time_grid();
    }
    else
    {
        compute_txw();

        const auto D = equation_->dim();
        const auto H = rff_.hidden_dim();

        y0_ = torch::randn({1}, torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(device_));

        alpha_ = torch::randn({D, H}, torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(device_)) * config_.solver_config.alpha_init_scale;
    }

    if (!is_linear_)
    {
        compute_H(t_, x_);
    }
}

/* Options
 * set the initial $y_0$, $alpha$, and $lambda$. */

RFMSolver& RFMSolver::options(
    const std::optional<torch::Tensor>& y0,
    const std::optional<torch::Tensor>& alpha,
    const std::optional<float> lambda
)
{
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

RFMSolver& RFMSolver::linear_options(const LinearSolverOptions& options)
{
    linear_solver_options_ = options;
    return *this;
}

/* Solver
 * `Solver` directs linear and nonlinear problems to different main solver functions. */

std::tuple<torch::Tensor, torch::Tensor, float> RFMSolver::solve(const bool output_log) const
{
    if (is_linear_) return solve_linear();
    return solve_nonlinear(output_log);
}

float RFMSolver::test(const torch::Tensor& y0, const torch::Tensor& alpha) const
{
    torch::NoGradGuard no_grad;

    const int64_t S = config_.solver_config.sample_size;
    const int64_t batch_size = linear_solver_options_.qr_batch_size > 0
        ? linear_solver_options_.qr_batch_size
        : S;

    double squared_error_sum = 0.0;
    int64_t residual_count = 0;
    for (int64_t row_begin = 0; row_begin < S; row_begin += batch_size)
    {
        const int64_t current_batch_size = std::min(batch_size, S - row_begin);
        const auto [batch_squared_error, batch_count] = test_batch(
            y0,
            alpha,
            current_batch_size
        );
        squared_error_sum += batch_squared_error;
        residual_count += batch_count;
    }

    return static_cast<float>(std::sqrt(squared_error_sum / static_cast<double>(residual_count)));
}

std::pair<double, int64_t> RFMSolver::test_batch(
    const torch::Tensor& y0,
    const torch::Tensor& alpha,
    const int64_t batch_size
) const
{
    using namespace torch::indexing;

    const int64_t T = config_.eqn_config.num_time_intervals;
    const int64_t D = equation_->dim();
    const int64_t Hdim = rff_.hidden_dim();
    const float dt = equation_->delta_t();

    TORCH_CHECK(alpha.numel() == D * Hdim, "alpha must have ", D * Hdim, " elements, got ", alpha.numel());

    const auto y0_eval = y0.to(device_).reshape({1});
    const auto alpha_eval = alpha.to(device_).reshape({D, Hdim}).contiguous();

    const auto [dw_sample, x_sample] = equation_->sample(batch_size);
    const auto dw_eval = dw_sample.to(device_).contiguous();
    const auto x_all = x_sample.to(device_).permute({0, 2, 1}).contiguous();

    const auto x_eval = x_all.index({
        Slice(),
        Slice(0, -1),
        Slice()
    }).unsqueeze(2).contiguous(); // (S, T, 1, D)

    const auto x_end_eval = x_all.index({
        Slice(),
        Slice(-1, None),
        Slice()
    }).unsqueeze(2).contiguous(); // (S, 1, 1, D)

    const auto t = t_.index({Slice(0, batch_size), Slice(), Slice(), Slice()}).contiguous();
    const auto t_end = t_end_.index({Slice(0, batch_size), Slice(), Slice(), Slice()}).contiguous();

    auto y = y0_eval.reshape({1, 1, 1, 1}).expand({batch_size, 1, 1, 1});
    const auto H_eval = rff_.phi(t, x_eval);
    const auto z_all = torch::matmul(
        H_eval.squeeze(-1).contiguous(),
        alpha_eval.transpose(0, 1)
    ).unsqueeze(2).contiguous(); // (S, T, 1, D)
    const auto dw_all = dw_eval.permute({0, 2, 1}).unsqueeze(2).contiguous(); // (S, T, 1, D)

    for (int64_t k = 0; k < T; ++k)
    {
        const auto t_k = t.index({Slice(), Slice(k, k + 1), Slice(), Slice()});
        const auto x_k = x_eval.index({Slice(), Slice(k, k + 1), Slice(), Slice()});
        const auto z_k = z_all.index({Slice(), Slice(k, k + 1), Slice(), Slice()});
        const auto dw_k = dw_all.index({Slice(), Slice(k, k + 1), Slice(), Slice()});

        const auto f_k = equation_->f(t_k, x_k, y, z_k);
        const auto martingale = torch::sum(dw_k * z_k, -1, true);

        TORCH_CHECK(
            f_k.sizes() == y.sizes(),
            "equation_->f must return shape ", y.sizes(), ", but got ", f_k.sizes()
        );

        y = y - dt * f_k + martingale;
    }

    const auto g_terminal = equation_->g(t_end, x_end_eval);
    TORCH_CHECK(
        g_terminal.sizes() == y.sizes(),
        "equation_->g must return shape ", y.sizes(), ", but got ", g_terminal.sizes()
    );

    const auto residual = y - g_terminal;
    return {
        residual.pow(2).sum().template item<double>(),
        residual.numel()
    };
}

std::tuple<torch::Tensor, torch::Tensor, float> RFMSolver::solve_linear() const
{
    auto options = linear_solver_options_;
    options.ridge_lambda = config_.solver_config.initial_lambda;

    if (options.solver_type == LinearSolverType::BatchedQR)
    {
        return solve_linear_batched_qr(options);
    }

    const auto [A, B] = compute_linear_coef();

    const auto [y0, alpha, rmse] = solve_linear_least_squares(
        A,
        B,
        config_.eqn_config.dimension,
        config_.solver_config.hidden_dim,
        options
    );

    return {y0, alpha, rmse};
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
    const int64_t S = config_.solver_config.sample_size;
    const int64_t batch_size = linear_solver_options_.qr_batch_size > 0
        ? linear_solver_options_.qr_batch_size
        : S;

    if (batch_size >= S)
    {
        return compute_linear_coef_batch(0, S);
    }

    std::vector<torch::Tensor> A_batches;
    std::vector<torch::Tensor> B_batches;
    A_batches.reserve((S + batch_size - 1) / batch_size);
    B_batches.reserve(A_batches.capacity());

    for (int64_t row_begin = 0; row_begin < S; row_begin += batch_size)
    {
        const int64_t row_end = std::min(row_begin + batch_size, S);
        const auto [A_batch, B_batch] = compute_linear_coef_batch(row_begin, row_end);
        A_batches.push_back(A_batch);
        B_batches.push_back(B_batch);
    }

    return {
        torch::cat(A_batches, 0).contiguous(),
        torch::cat(B_batches, 0).contiguous()
    };
}

std::pair<const torch::Tensor, const torch::Tensor> RFMSolver::compute_linear_coef_batch(
    const int64_t row_begin,
    const int64_t row_end
) const
{
    const int64_t S = config_.solver_config.sample_size;
    const int64_t T = config_.eqn_config.num_time_intervals;
    const int64_t D = equation_->dim();
    const int64_t Hdim = config_.solver_config.hidden_dim;
    const float dt = equation_->delta_t();
    const int64_t batch_size = row_end - row_begin;

    TORCH_CHECK(0 <= row_begin && row_begin < row_end && row_end <= S,
        "invalid linear coefficient row range [", row_begin, ", ", row_end, ") for S=", S);

    const auto [dw_sample, x_sample] = equation_->sample(batch_size);
    const auto dw_batch = dw_sample.to(device_).contiguous();
    const auto x_all = x_sample.to(device_).permute({0, 2, 1}).contiguous(); // (B, T+1, D)
    const auto x = x_all.index({
        torch::indexing::Slice(),
        torch::indexing::Slice(0, -1),
        torch::indexing::Slice()
    }).unsqueeze(2).contiguous(); // (B, T, 1, D)
    const auto x_end = x_all.index({
        torch::indexing::Slice(),
        torch::indexing::Slice(-1, torch::indexing::None),
        torch::indexing::Slice()
    }).unsqueeze(2).contiguous(); // (B, 1, 1, D)

    const auto t = t_.index({
        torch::indexing::Slice(0, batch_size),
        torch::indexing::Slice(),
        torch::indexing::Slice(),
        torch::indexing::Slice()
    }).contiguous();
    const auto t_end = t_end_.index({
        torch::indexing::Slice(0, batch_size),
        torch::indexing::Slice(),
        torch::indexing::Slice(),
        torch::indexing::Slice()
    }).contiguous();

    const auto L = equation_->coef().L(t, x).squeeze(-1).squeeze(-1).contiguous(); // (B, T)
    const auto M = equation_->coef().M(t, x).squeeze(2).contiguous();              // (B, T, D)
    const auto N = equation_->coef().N(t, x).squeeze(-1).squeeze(-1).contiguous(); // (B, T)
    const auto H = rff_.phi(t, x).squeeze(-1).contiguous();                       // (B, T, H)
    const auto dW = dw_batch.permute({0, 2, 1}).contiguous();                     // (B, T, D)

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
    auto coef_y0 = a.prod(1, true); // (B, 1)

    // 矩阵第二块: alpha 系数
    // weighted_xi: (S, T, D)
    const auto weighted_xi = xi * weights.unsqueeze(-1);

    // coef_alpha[s] = weighted_xi[s]^T @ H[s] -> (D, H)
    auto coef_alpha = torch::bmm(
        weighted_xi.transpose(1, 2).contiguous(), // (B, D, T)
        H                                          // (B, T, H)
    );                                             // (B, D, H)

    coef_alpha = coef_alpha.reshape({batch_size, D * Hdim}); // (B, D*H)

    // 拼接设计矩阵
    const auto A = torch::cat({coef_y0, coef_alpha}, 1).contiguous(); // (B, 1 + D*H)

    // 右端项
    const auto constant_part = (weights * c).sum(1, true); // (B, 1)
    const auto g_XN = equation_->g(t_end, x_end).reshape({batch_size, 1}).to(device_);

    const auto B = g_XN - constant_part; // (B, 1)

    TORCH_CHECK(
        A.device().type() == device_.type() &&
        B.device().type() == device_.type(),
        "A, B must be on ", device_.type(), ", but got ", A.device().type(), " & ", B.device().type())

    return {A, B};
}

std::tuple<torch::Tensor, torch::Tensor, float> RFMSolver::solve_linear_batched_qr(
    const LinearSolverOptions& options
) const
{
    const int64_t S = config_.solver_config.sample_size;
    const int64_t D = config_.eqn_config.dimension;
    const int64_t Hdim = config_.solver_config.hidden_dim;
    const int64_t parameter_count = 1 + D * Hdim;

    TORCH_CHECK(options.qr_batch_size > 0, "qr_batch_size must be positive");

    bool initialized = false;
    int64_t pending_rows = 0;
    std::vector<torch::Tensor> pending_A;
    std::vector<torch::Tensor> pending_B;
    torch::Tensor R;
    torch::Tensor rhs;

    for (int64_t row_begin = 0; row_begin < S; row_begin += options.qr_batch_size)
    {
        const int64_t row_end = std::min(row_begin + options.qr_batch_size, S);
        const auto [A_batch, B_batch] = compute_linear_coef_batch(row_begin, row_end);

        if (!initialized)
        {
            pending_A.push_back(A_batch);
            pending_B.push_back(B_batch);
            pending_rows += A_batch.size(0);

            if (pending_rows < parameter_count)
            {
                continue;
            }

            const auto A_init = torch::cat(pending_A, 0).contiguous();
            const auto B_init = torch::cat(pending_B, 0).contiguous();
            std::tie(R, rhs) = reduce_linear_qr_libtorch(
                A_init,
                B_init
            );
            rhs = rhs.reshape({parameter_count, 1}).contiguous();
            initialized = true;
            pending_A.clear();
            pending_B.clear();
            continue;
        }

        const auto stacked_A = torch::cat({R, A_batch}, 0).contiguous();
        const auto stacked_B = torch::cat({rhs, B_batch}, 0).contiguous();
        std::tie(R, rhs) = reduce_linear_qr_libtorch(
            stacked_A,
            stacked_B
        );
        rhs = rhs.reshape({parameter_count, 1}).contiguous();
    }

    TORCH_CHECK(initialized, "sample_size must be at least 1 + dimension * hidden_dim for batched QR");

    const auto x = torch::linalg_solve_triangular(R, rhs, true).reshape({parameter_count, 1});

    double squared_error_sum = 0.0;
    int64_t residual_count = 0;
    for (int64_t row_begin = 0; row_begin < S; row_begin += options.qr_batch_size)
    {
        const int64_t row_end = std::min(row_begin + options.qr_batch_size, S);
        const auto [A_batch, B_batch] = compute_linear_coef_batch(row_begin, row_end);
        const auto residual = torch::matmul(A_batch, x) - B_batch;
        squared_error_sum += residual.pow(2).sum().item<double>();
        residual_count += residual.numel();
    }

    const float rmse = static_cast<float>(std::sqrt(squared_error_sum / static_cast<double>(residual_count)));
    return split_linear_solution(x, D, Hdim, rmse);
}

std::tuple<torch::Tensor, torch::Tensor, float> RFMSolver::solve_nonlinear_levenberg_marquardt(
    const torch::Tensor &y0, const torch::Tensor &alpha, const float lambda, const bool output_log) const
{
    const int64_t max_iters = config_.solver_config.num_iterations;

    torch::Tensor theta = solver_utils::pack_nonlinear_parameters(y0, alpha).detach().clone().to(device_);
    float damping = lambda;
    float final_error = 0.0f;

    for (int64_t iter = 0; iter < max_iters; ++iter)
    {
        const auto&[
            min_lambda,
            max_lambda,
            lambda_decrease,
            lambda_increase,
            error_tol,
            step_tol,
            max_retries
        ] = config_.solver_config.nonlinear;

        const auto [residual_raw, jacobian] =
            compute_nonlinear_terminal_residual_and_jacobian(theta);
        const auto residual = residual_raw.reshape({-1});
        const float curr_loss = 0.5f * residual.pow(2).sum().item<float>();
        const float curr_error = std::sqrt(residual.pow(2).mean().item<float>());

        bool accepted = false;
        torch::Tensor accepted_theta;
        float accepted_error = 0;
        float accepted_step_norm = 0.0f;

        for (int64_t retry = 0; retry <= max_retries; ++retry)
        {
            const auto delta = solver_utils::solve_lm_step(jacobian, residual, damping);
            const auto step_norm = delta.norm().item<float>();

            const auto trial_theta = (theta + delta).detach();
            const auto trial_residual = compute_nonlinear_terminal_residual(trial_theta).reshape({-1});
            const float trial_loss = 0.5f * trial_residual.pow(2).sum().item<float>();
            const float trial_error = std::sqrt(trial_residual.pow(2).mean().item<float>());
            accepted = trial_loss < curr_loss;

            if (output_log)
            {
                std::cout
                    << "[LM] iter=" << iter
                    << " retry=" << retry
                    << " loss=" << curr_loss
                    << " error=" << curr_error
                    << " trial_error=" << trial_error
                    << " lambda=" << damping
                    << " step_norm=" << step_norm
                    << " accepted=" << std::boolalpha << accepted
                    << " y_0=" << trial_theta.index({0}).item<float>()
                    << std::noboolalpha
                    << std::endl;
            }

            if (accepted)
            {
                accepted_theta = trial_theta;
                accepted_error = trial_error;
                accepted_step_norm = step_norm;
                break;
            }

            if (retry < max_retries)
            {
                damping = std::min(max_lambda, damping * lambda_increase);
            }
        }

        if (!accepted)
        {
            if (output_log)
            {
                std::cout
                    << "[LM] stop: no accepted step after "
                    << max_retries + 1
                    << " attempts at iter=" << iter
                    << std::endl;
            }
            break;
        }

        theta = accepted_theta;
        damping = std::max(min_lambda, damping * lambda_decrease);
        final_error = accepted_error;

        if (final_error <= error_tol || accepted_step_norm <= step_tol)
        {
            break;
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
    const int64_t S = config_.solver_config.sample_size;
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
    const auto g_terminal = equation_->g(t_end_, x_end_);

    TORCH_CHECK(
        g_terminal.sizes() == y_terminal.sizes(),
        "equation_->g must return shape ", y_terminal.sizes(), ", but got ", g_terminal.sizes()
    );

    const auto residual = y_terminal - g_terminal;
    return residual.reshape({S, 1}).contiguous();
}

std::pair<torch::Tensor, torch::Tensor> RFMSolver::compute_nonlinear_terminal_residual_and_jacobian(
    const torch::Tensor& theta
) const
{
    const int64_t D = equation_->dim();
    const int64_t Hdim = rff_.hidden_dim();
    const int64_t expected_size = 1 + D * Hdim;

    TORCH_CHECK(
        theta.dim() == 1 && theta.size(0) == expected_size,
        "theta must have shape (", expected_size, "), but got ", theta.sizes()
    );

    if (equation_->has_analytic_jacobian())
    {
        const auto y0 = theta.index({0}).reshape({1});
        const auto alpha = theta.index({
            torch::indexing::Slice(1, torch::indexing::None)
        }).reshape({D, Hdim}).contiguous();

        auto [residual, jacobian] = equation_->terminal_residual_and_jacobian(
            t_, t_end_, x_, x_end_, dw_, H_, y0, alpha
        );

        TORCH_CHECK(residual.dim() == 2 && residual.size(1) == 1,
            "analytic residual must have shape (S, 1), but got ", residual.sizes());
        TORCH_CHECK(jacobian.dim() == 2 && jacobian.size(0) == residual.size(0) &&
            jacobian.size(1) == expected_size,
            "analytic Jacobian must have shape (S, ", expected_size, "), but got ", jacobian.sizes());

        return {residual.contiguous(), jacobian.contiguous()};
    }

    const auto theta_with_grad = theta.detach().clone().requires_grad_(true);
    const auto residual = compute_nonlinear_terminal_residual(theta_with_grad).reshape({-1});
    auto jacobian = solver_utils::compute_nonlinear_jacobian(residual, theta_with_grad);

    return {residual.reshape({-1, 1}).contiguous(), jacobian};
}

torch::Tensor RFMSolver::forward_nonlinear_terminal_y(
    const torch::Tensor& y0,
    const torch::Tensor& alpha
) const
{
    using namespace torch::indexing;

    const int64_t S = config_.solver_config.sample_size;
    const int64_t T = config_.eqn_config.num_time_intervals;
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

        const auto f_k = equation_->f(t_k, x_k, y, z_k);
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

void RFMSolver::compute_time_grid()
{
    const double total_time = config_.eqn_config.total_time;
    const int64_t T = config_.eqn_config.num_time_intervals;
    const int64_t S = config_.solver_config.sample_size;

    const auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    const auto t_full = torch::linspace(0, total_time, T + 1, opts);

    const auto t_base = t_full.slice(0, 0, T).reshape({1, T, 1, 1});
    t_ = t_base.expand({S, T, 1, 1}).contiguous();

    const auto t_end_base = t_full.slice(0, T, T + 1).reshape({1, 1, 1, 1});
    t_end_ = t_end_base.expand({S, 1, 1, 1}).contiguous();
}

void RFMSolver::compute_txw()
{
    compute_time_grid();

    const int64_t S = config_.solver_config.sample_size;

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
        result.size(0) == config_.solver_config.sample_size &&
        result.size(1) == config_.eqn_config.num_time_intervals &&
        result.size(2) == 1 &&
        result.size(3) == 1,
        "Invalid shape for L(t, x). Expected (",
        config_.solver_config.sample_size, ", ",
        config_.eqn_config.num_time_intervals, ", 1, 1), but got ",
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
        result.size(0) == config_.solver_config.sample_size &&
        result.size(1) == config_.eqn_config.num_time_intervals &&
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
        result.size(2) == config_.solver_config.hidden_dim &&
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
        t.size(1) == config_.eqn_config.num_time_intervals &&
        t.size(2) == 1 &&
        t.size(3) == 1,
        "Invalid shape for t. Expected (",
        x.size(0), " or 1, ",
        config_.eqn_config.num_time_intervals,
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
        x.size(0) == config_.solver_config.sample_size &&
        x.size(1) == config_.eqn_config.num_time_intervals &&
        x.size(2) == 1 &&
        x.size(3) == config_.eqn_config.dimension,
        "Invalid shape for x. Expected (",
        config_.solver_config.sample_size, ", ",
        config_.eqn_config.num_time_intervals,
        ", 1, ", config_.eqn_config.dimension,
        "), but got ", x.sizes()
    );

    TORCH_CHECK(
        x.device().type() == device_.type() &&
        t.device().type() == device_.type(),
        "x, t must be on ", device_.type(), ", but got ", x.device().type(), " & ", t.device().type()
    );
}
